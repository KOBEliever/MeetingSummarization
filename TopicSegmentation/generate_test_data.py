import json
import os


def generate_test_data_from_file(file):
    seg_info = json.load(open(file, "r", encoding="utf-8"))
    all_context = {}
    with open("../dataset_1228/overall_context.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            av_num = str(jo["av_num"])
            context = jo["context"]
            all_context[av_num] = context

    fw = open(f"seg_files/{file.split('/')[-1].replace('.json', '_seg.txt')}", "w", encoding="utf-8")

    for k, v in seg_info.items():
        context = all_context[k]
        pred_seg_list = v["preds"]
        assert len(pred_seg_list) == len(context)

        _tmp_seg_context = []
        for i in range(len(pred_seg_list)):
            if pred_seg_list[i] == "0":
                _tmp_seg_context.append(context[i])
            else:
                _tmp_seg_context.append(context[i])
                if sum([len("".join(_)) for _ in _tmp_seg_context]) > 512:
                    fw.write(json.dumps({"av_num": k, "context": _tmp_seg_context, "discussion": "discussion", "agenda": "agenda"},
                                        ensure_ascii=False) + "\n")
                    _tmp_seg_context = []

        if len(_tmp_seg_context) > 0 and sum([len(_) for _ in _tmp_seg_context]) > 512:
            fw.write(json.dumps({"av_num": k, "context": _tmp_seg_context, "discussion": "discussion", "agenda": "agenda"},
                                ensure_ascii=False) + "\n")


if __name__ == '__main__':
    generate_test_data_from_file("output/output_8-64-bart.json")
    generate_test_data_from_file("output/output_focal-12-32-bert.json")
    generate_test_data_from_file("output/output_12-32-bert.json")
    generate_test_data_from_file("output/output_focal-10-32-bert.json")
    generate_test_data_from_file("output/output_focal-8-64-bert.json")
