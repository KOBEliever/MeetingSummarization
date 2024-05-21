import json
import os


def obtain_sg_data(file):
    sg_dict = {}
    with open(file, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            meeting_id = jo["id"].split("_")[0]
            av_num = jo["av_num"]
            context = jo["context"]
            agenda = jo["agenda"]
            discussion = jo["discussion"]

            if av_num not in sg_dict:
                _tmp_dict = {"id": meeting_id, "av_num": av_num, "eos_pos": [len(context)],
                             "summary": [f"{agenda}:{discussion}"], "context": context}
                sg_dict[av_num] = _tmp_dict
            else:
                sg_dict[av_num]["context"].extend(context)
                sg_dict[av_num]["eos_pos"].append(len(sg_dict[av_num]["context"]))
                sg_dict[av_num]["summary"].append(f"{agenda}:{discussion}")

    with open(f"sg_{file.split('/')[-1].rstrip('.txt')}.txt", "w", encoding="utf-8") as fw:
        for k, v in sg_dict.items():
            fw.write(json.dumps(v, ensure_ascii=False) + "\n")


def segment_long_context(file, max_turn_num=50, is_inference=False):
    if is_inference:
        path_prefix = "inference"
    else:
        if not os.path.exists(str(max_turn_num)):
            os.mkdir(str(max_turn_num))
        path_prefix = str(max_turn_num)

    with open(file, "r", encoding="utf-8") as fr, open(f"{path_prefix}/" + file.replace("sg_", f"sg_{max_turn_num}_"), "w", encoding="utf-8") as fw:
        for line in fr:
            jo = json.loads(line.strip())
            sid, av_num, eos_pos = jo["id"], jo["av_num"], jo["eos_pos"]
            context, summary = jo["context"], jo["summary"]
            eos_pos = [_ - 1 for _ in eos_pos]
            i = 0
            split_turns = []
            while len(context) > max_turn_num:
                new_id = f"{sid}_{i}"
                new_context = context[:max_turn_num]
                new_eos_pos = [e - max_turn_num * i for e in eos_pos if e < max_turn_num * (i + 1)]
                new_summary = summary[:len(new_eos_pos)]
                eos_pos = eos_pos[len(new_eos_pos):]
                summary = summary[len(new_eos_pos):]
                context = context[max_turn_num:]
                split_turns.append({"id": new_id, "av_num": av_num, "eos_pos": new_eos_pos,
                                    "summary": new_summary, "context": new_context})
                i += 1
            split_turns.append({"id": f"{sid}_{i}", "av_num": av_num, "eos_pos": [e-max_turn_num*i for e in eos_pos],
                                "summary": summary, "context": context})
            for t in split_turns:
                if len(t["eos_pos"]) > 0 or is_inference:
                    fw.write(json.dumps(t, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    # obtain_sg_data("../../dataset_1228/short_train.txt")
    # obtain_sg_data("../../dataset_1228/short_dev.txt")
    for i in [5, 8, 10, 12, 15, 20, 50]:
        segment_long_context("sg_short_dev.txt", max_turn_num=i)
        segment_long_context("sg_short_test.txt", max_turn_num=i)
        segment_long_context("sg_short_test.txt", max_turn_num=i, is_inference=True)
        segment_long_context("sg_short_train.txt", max_turn_num=i)
