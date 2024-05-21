import json
import random
from tqdm import tqdm
from nltk.metrics.segmentation import windowdiff, pk


def extract_seg_labels(file):
    labels = []
    jo = json.load(open(file, "r", encoding="utf-8"))
    for k, v in jo.items():
        labels.append(v["label"])
    fw = open("labels.txt", "w", encoding="utf-8")
    fw.write("\n".join(labels))


def random_baseline(iter_num=50):
    seed = 42
    random.seed(seed)
    labels = open("labels.txt", "r", encoding="utf-8").readlines()
    wd_score_all, pk_score_all = [], []
    for i in tqdm(range(iter_num)):
        turn_num = random.choice(list(range(3, 15)))
        wd_score_case, pk_score_case = [], []
        for line in labels:
            line = line.strip()
            idx_list = list(range(len(line)))
            random.shuffle(idx_list)
            sel_idx = idx_list[:turn_num]
            pred = ["0"] * len(line)
            for i in sel_idx:
                pred[i] = '1'
            pred = "".join(pred)
            wd_score = windowdiff(pred, line, k=3, weighted=True)
            pk_score = pk(line, pred, k=3)
            wd_score_case.append(wd_score)
            pk_score_case.append(pk_score)
        wd_score_all.append(sum(wd_score_case)/len(wd_score_case))
        pk_score_all.append(sum(pk_score_case)/len(pk_score_case))
    wd_avg = round(sum(wd_score_all) / len(wd_score_all), 4)
    pk_avg = round(sum(pk_score_all) / len(pk_score_all), 4)
    print(f"avg wd: {wd_avg}, pk avg: {pk_avg}")


def even_baseline(iter_num=50):
    seed = 42
    random.seed(seed)
    labels = open("labels.txt", "r", encoding="utf-8").readlines()
    wd_score_all, pk_score_all = [], []
    for i in tqdm(range(iter_num)):
        turn_num = random.choice(list(range(3, 15)))
        wd_score_case, pk_score_case = [], []
        for line in labels:
            line = line.strip()
            pred = ["0"] * len(line)
            even_idx = max(2, int(len(line) // turn_num))
            for idx in range(len(line)):
                if idx % even_idx == 0:
                    pred[idx] = "1"
            pred = "".join(pred)
            wd_score = windowdiff(pred, line, k=3, weighted=True)
            pk_score = pk(line, pred, k=3)
            wd_score_case.append(wd_score)
            pk_score_case.append(pk_score)
        wd_score_all.append(sum(wd_score_case) / len(wd_score_case))
        pk_score_all.append(sum(pk_score_case) / len(pk_score_case))
    wd_avg = round(sum(wd_score_all) / len(wd_score_all), 4)
    pk_avg = round(sum(pk_score_all) / len(pk_score_all), 4)
    print(f"avg wd: {wd_avg}, pk avg: {pk_avg}")


if __name__ == '__main__':
    # extract_seg_labels("../output/output_15-128-bert.json")
    even_baseline()
