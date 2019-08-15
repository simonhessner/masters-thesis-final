import argparse
import logging
import os
from glob import glob
import importlib
import sys
import csv
import traceback
import time
from multiprocessing import Process, Queue
from common.util import mkdir_if_not_exists, print_info, get_human_readable_time, LossException
import json

#DEFAULT_DATASET = "/media/bighdd5/simon/projects/data/datasets/hourglass_samples_with_afw_without_multipie.h5"

# TODO improve logging, use Lock to ensure only one process prints to stdout?
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
# https://docs.python.org/3/howto/logging-cookbook.html

ALL_WORK_DONE_TOKEN = "__NOTHING_LEFT_TO_DO__" # Used to mark queue end


class Worker(object):
    def __init__(self, gpu_id, sub_gpu_id, config_queue, model_dir, output_dir, data, result_queue):
        self.gpu_id = int(gpu_id)
        self.sub_gpu_id = int(sub_gpu_id)
        self.config_queue = config_queue
        self.trainer_module = importlib.import_module("%s.main" % model_dir)
        self.output_dir = output_dir
        self.result_queue = result_queue
        self.data = data

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print_info("Created folder '%s'" % self.output_dir)

    def work(self):
        while True:
            configfile = self.config_queue.get()
            if configfile == ALL_WORK_DONE_TOKEN:
                msg = "GPU %d.%d found an empty queue => Terminate" % (self.gpu_id, self.sub_gpu_id)
                print_info(msg)
                logging.info(msg)
                return

            print_info("[qsize=%d] GPU %d.%d -> %s" % (self.config_queue.qsize(), self.gpu_id, self.sub_gpu_id, configfile))

            time.sleep((self.gpu_id + self.sub_gpu_id + 100) / 1000)

            try:
                model_trainer = self.trainer_module.ModelTrainer(config=configfile,
                                                                 output_dir=self.output_dir,
                                                                 gpu_id=self.gpu_id,
                                                                 sub_gpu_id=self.sub_gpu_id,
                                                                 data=self.data,
                                                                 gridsearch=True)
                results = model_trainer.run()
                self.result_queue.put(results)
                msg = "GPU %d.%d finished %s" % (self.gpu_id, self.sub_gpu_id, configfile)
                print_info(msg)
                logging.info(msg)

            except Exception as e:
                msg = "Config '%s' failed on GPU %d.%d: \n>>> %s" % (configfile, self.gpu_id, self.sub_gpu_id, str(e))
                print_info(msg)
                logging.error(msg)
                if not isinstance(e, LossException):
                    traceback.print_exc()

                try:
                    fn = os.path.join(self.output_dir, "failed.txt")
                    with open(fn, "a") as f:
                        f.write("[%s] %s : %s\n\n" % (get_human_readable_time(), configfile, msg))
                except Exception as e2:
                    msg = "Failure could not be written to failed.txt: %s" % str(e)
                    print_info(msg)
                    logging.error(msg)

            time.sleep(1)  # Wait in case GPU needs time to release memory

    @staticmethod
    def run_worker(gpu_id, sub_gpu_id, config_queue, model_dir, output_dir, data, result_queue):
        """
        This static helper method is to ensure that is no reference to each worker in the parent process.
        This is just to avoid any side effects of using PyTorch and other libs in parallel (these side effects
        occured when using Threads instead of Processes and caused DataLoader to freeze randomly)
        """
        worker = Worker(gpu_id, sub_gpu_id, config_queue, model_dir, output_dir, data, result_queue)
        worker.work()


class ResultHandler(object):
    def __init__(self, queue, result_file, done_list_file):
        self.queue = queue
        self.result_file = result_file
        self.done_list_file = done_list_file

    def work(self):
        with open(self.result_file, 'a', newline='') as csvfile, \
             open(self.done_list_file, 'a') as done_list:
            writer = None

            while True:
                result = self.queue.get()
                if result == ALL_WORK_DONE_TOKEN:
                    msg = "Result writer process found an empty queue => Terminate"
                    print_info(msg)
                    logging.info(msg)
                    return

                if not writer:
                    # This is a little workaround to the problem that it must be known in advance which fields
                    # will be written to the file. The first result in the queue defines the set of fieldnames
                    # that will be written to the header. It is assumed that every result contains the same set
                    # of keys. If new keys are added to new results, an exception will be thrown.
                    ks = [k for k in result.keys() if k != "metrics_log"]
                    writer = csv.DictWriter(csvfile, fieldnames=ks)
                    writer.writeheader()

                if "metrics_log" in result:
                    metrics_log = result["metrics_log"]
                    result = {k:v for k,v in result.items() if k != "metrics_log"}
                    config_id = result["config_id"]

                    directory, _ = os.path.split(self.result_file)
                    target = os.path.join(directory, "results", "%d_metrics.log" % config_id)
                    json.dump(metrics_log, open(target, "w"))

                writer.writerow(result)
                csvfile.flush()  # needed to ensure that nothing is lost in case of crashes

                # Write the path to the config file that has just been finished to a file that can be used the next
                # time to skip already completed configs.
                done_list.write("%s\n" % result["config_file"])  # TODO only write config id
                done_list.flush()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Perform grid search on a model in MODEL_DIR using hyperparameter"
                                                 "configurations from CONFIG_DIR and write results to OUTPUT_DIR")
    parser.add_argument('model_dir',
                        type=str,
                        help='directory of model that contains a main.py')

    parser.add_argument('config_dir',
                        type=str,
                        help='directory of commands')

    parser.add_argument('output_dir',
                        type=str,
                        help='where to store plots, trained models, results, etc')

    parser.add_argument('dataset',
                        type=str,
                        help="Source data to train model")

    parser.add_argument('--no_skip',
                        action="store_true",
                        help="Ignore items in done_configs.txt and train them again (if this flag is not set, configs "
                             "listed in done_configs.txt will be skipped. Useful when training should be resumed). "
                             "Existing models, plots, results/*.txt will be overridden")

    parser.add_argument('--gpus',
                        type=int,
                        default=[0, 1, 2, 3],
                        nargs="+",
                        help='IDs of GPUs to use, separated by spaces')

    parser.add_argument('--workers_per_gpu',
                        type=int,
                        default=1,
                        help="For small models, run multiple models on one GPU")

    opt = parser.parse_args()

    output_dir = os.path.abspath(opt.output_dir)
    mkdir_if_not_exists(output_dir)
    logfile = os.path.join(output_dir, "trainer.log")

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] <%(levelname)s> %(message)s',
                        filename=logfile, filemode='a')
    logging.info("-------- trainer.py started ------------")

    print("Training '%s' with confs '%s' on GPUs %s" % (opt.model_dir, opt.config_dir, ",".join(map(str, opt.gpus))))

    # Needed to enable models to import relatively to their own folder (they do not have to know about trainer.py)
    sys.path.append(os.path.join(os.path.dirname(__file__), opt.model_dir))

    skip_list_file = os.path.join(output_dir, "done_configs.txt")
    skip_configfiles = []
    if not opt.no_skip and os.path.exists(skip_list_file) and os.path.isfile(skip_list_file):
        with open(skip_list_file, 'r') as f:
            skip_configfiles = f.read().splitlines()

    config_queue = Queue()
    for configfile in glob(os.path.join(opt.config_dir, "*.json")): # TODO only check for ID and not full path
        if configfile in skip_configfiles:
            print("Skip %s" % configfile)
        else:
            config_queue.put(configfile)

    results_csv = os.path.join(opt.output_dir, "results_0.csv")
    c = 1
    while os.path.exists(results_csv) and os.path.isfile(results_csv):
        print("%s exists" % results_csv)
        results_csv = os.path.join(opt.output_dir, "results_%d.csv" % c)
        c += 1
    print("results file: %s" % results_csv)

    result_queue = Queue()
    result_handler = ResultHandler(result_queue, results_csv, skip_list_file)
    result_process = Process(name="result_handler", target=result_handler.work)
    result_process.daemon = True
    result_process.start()

    processes = []
    for gpu_id in opt.gpus:
        for sub_gpu_id in range(opt.workers_per_gpu):
            config_queue.put(ALL_WORK_DONE_TOKEN) # To signal the workers that they can stop, put this num_worker times at the end of the queue (so each process will read it exactly one time at the end)
            p = Process(name="gpu_%d.%d_worker" % (gpu_id, sub_gpu_id), target=Worker.run_worker, args=(gpu_id, sub_gpu_id, config_queue, opt.model_dir, output_dir, opt.dataset, result_queue))
            p.start()
            processes.append(p)
            print_info("Worker for GPU %d.%d started" % (gpu_id, sub_gpu_id))

    for p in processes:
        p.join()
    print("All jobs are done")

    result_queue.put(ALL_WORK_DONE_TOKEN) # To signal the process that it can stop
    result_process.join()
    print("All results written")

    logging.shutdown()


if __name__ == '__main__':
    main()