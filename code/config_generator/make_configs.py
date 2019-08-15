#!/usr/bin/env python3 

import json
import itertools
import os
import csv
import random
import math
from tqdm import tqdm
import yaml
import argparse
from common.util import mkdir_if_not_exists

parser = argparse.ArgumentParser(description='Generate gridsearch parameters.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 allow_abbrev=False) # For example, this will not accept --growth where --growth_rate is expected

parser.add_argument('template_file',
                    type=str,
                    help='YAML file that lists all possible values for all needed parameters')
parser.add_argument('output_directory',
                    type=str,
                    help='Configs will be saved in this directory')
parser.add_argument('--num_workers',
                    type=int,
                    default=5,
                    help='Split generated configs up in num_workers directories (useful if you want to use multiple machines)')
parser.add_argument('--stop_after',
                    type=int,
                    default=None,
                    help='Only create N configs (randomly sampled). Useful if there are too many combinations')
parser.add_argument('--count_only',
                    default=False,
                    action='store_true',
                    help='Will not create any files, just count combinations')

args = parser.parse_args()
options = yaml.safe_load(open(args.template_file, "r"))
model = options['model']
hyperparameters = options['hyperparameters']

print("Read %s" % args.template_file)
print(">> Model: %s" % model)
print(">> HYPERPARAMETERS:")

max_key_len = max(map(len, hyperparameters.keys()))
for k,v in hyperparameters.items():
    print("%s %s ->  %s" % (k, " "*(max_key_len-len(k)), str(v)))

root_folder = args.output_directory
mkdir_if_not_exists(root_folder)
print("\nWriting everything to '%s'" % root_folder)

num_workers=args.num_workers

extra_fields = ['model', 'config_id']

hparam_names = list(hyperparameters.keys())
hparam_values = list(hyperparameters.values())

combinations = [(i, combi) for i, combi in enumerate(itertools.product(*hparam_values))]
if args.count_only:
    print(len(combinations), "combinations")
    exit()

random.shuffle(combinations)
chunksize = math.ceil(len(combinations) / num_workers)
print("Generating %d configuration files in %d folders (~%d per folder)" % (len(combinations), num_workers, chunksize))


with open(os.path.join(root_folder, 'config_list_{}.txt'.format(model)), 'w') as overview_file:
    writer = csv.DictWriter(overview_file, fieldnames=extra_fields+hparam_names)
    writer.writeheader()

    combinations = list(enumerate(combinations))
    if args.stop_after:
        print("Will only generate %d/%d combinations" % (args.stop_after, len(combinations)))
        combinations = combinations[:args.stop_after]

    for counter, (config_id, combi) in tqdm(combinations):
        worker_dir = os.path.join(root_folder, str(counter // chunksize))
        if not os.path.exists(worker_dir):
            os.mkdir(worker_dir)
        configfile = os.path.join(worker_dir,"config_%d.json" % config_id)

        ops = {hparam_name : combi[j] for j, hparam_name in enumerate(hparam_names)}
        ops['model'] = model
        ops['config_id'] = config_id

        with open(configfile, 'w') as f:
            json.dump(ops, f, indent=4)
        writer.writerow(ops)