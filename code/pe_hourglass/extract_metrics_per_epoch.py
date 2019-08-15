import json
from glob import glob
from tqdm import tqdm
from csv import DictWriter


def get_performance_per_epoch(fn):
    with open(fn) as fp:
        d = json.load(fp)
        for epoch, metrics in enumerate(d["metric_histroy"]):
            yield {
                "config_id": d["config_id"],
                "epoch": epoch,
                "easy_49": metrics["easy_noutline"],
                "easy_68": metrics["easy_woutline"],
                "hard_49": metrics["hard_noutline"],
                "hard_68": metrics["hard_woutline"],
                "avg_49": (metrics["easy_noutline"] + metrics["hard_noutline"])/2,
                "avg_68": (metrics["easy_woutline"] + metrics["hard_woutline"])/2,
                "avg_all": (metrics["easy_noutline"] + metrics["hard_noutline"] + metrics["easy_woutline"] + metrics["hard_woutline"]) / 4
            }


writer = None

with open("/usr0/home/shessner/atlas_shared/results/thesis/hg/hg_arch_cmp/metrics_per_epoch.csv", "w") as csvfile:
    for f in tqdm(glob("/usr0/home/shessner/atlas_shared/results/thesis/hg/hg_arch_cmp/*/results/*.json")):
        for p in get_performance_per_epoch(f):
            if not writer:
                writer = DictWriter(csvfile, p.keys())
                writer.writeheader()
            writer.writerow(p)