
import logging
import os
import sys
import numpy as np
from datasets import load_dataset, set_caching_enabled
import json
import warnings

from transformers import (
    BertTokenizerFast,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from seg_utils import ModelArguments, DataTrainingArguments

from modeling_bart import BartForEncoderCls
from modeling_bert import BertForEncoderCls
from transformers import Trainer, TrainingArguments
from data_collator import DataCollatorForTokenClassification
from datasets.utils.download_manager import DownloadMode

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
set_caching_enabled(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess_function(example, tokenizer, max_length=256):

    inputs, attention_mask, position_ids, token_type_ids = [], [], [], []
    eos_ids, eos_attention_mask = [], []
    labels = []
    for i in range(len(example["context"])):
        if example["context"][i] is not None and example["eos_pos"][i] is not None:
            text_outputs = tokenizer(
                ["".join(c) for c in example["context"][i]],
                add_special_tokens=False,
                truncation=False,
            )
            text_input_ids = text_outputs["input_ids"]
            _tmp_input_ids, _tmp_position_ids, _tmp_eos_pos_ids, _tmp_token_type_ids = [], [], [], []
            input_ids_length = []
            for idx, text in enumerate(text_input_ids):
                if len(text) > max_length:
                    # text = text[:int(max_length//2)] + text[-int(max_length//2):]
                    text = text[-int(max_length-1):]
                text.insert(0, tokenizer.cls_token_id)
                _tmp_input_ids.extend(text)
                _tmp_token_type_ids.extend([idx % 2] * len(text))
                input_ids_length.append(len(text))
            total_length = sum(input_ids_length)
            _tmp_attention_mask = [1] * total_length
            acc_len = 0
            for cur_len in input_ids_length:
                # _tmp_mask = [0] * acc_len + [1] * cur_len + [0] * (total_length - acc_len - cur_len)
                # _tmp_mask = [1] * total_length
                # _tmp_attention_mask.extend([_tmp_mask] * cur_len)
                _tmp_position_ids.extend(list(range(acc_len, acc_len + cur_len)))
                acc_len += cur_len
                _tmp_eos_pos_ids.append(acc_len - 1)
            attention_mask.append(_tmp_attention_mask)
            inputs.append(_tmp_input_ids)
            position_ids.append(_tmp_position_ids)
            eos_ids.append(_tmp_eos_pos_ids)
            token_type_ids.append(_tmp_token_type_ids)
            utt_num = len(example["context"][i])
            eos_attention_mask.append([1 for _ in range(utt_num)])
            labels.append([int(_ in example["eos_pos"][i]) for _ in range(utt_num)])

    model_inputs = {
        "input_ids": inputs,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
        "eos_ids": eos_ids,
        "eos_attention_mask": eos_attention_mask,
        "labels": labels
    }

    return model_inputs


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    # 解析模型参数,数据参数,训练参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # 设置日志
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 设置输出 checkpoint
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    seed = training_args.seed + training_args.local_rank
    set_seed(seed)

    # 加载数据集
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # 加载模型(BARTenc, BERTenc),tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    if "bart" in model_args.model_name_or_path.lower():
        model = BartForEncoderCls.from_pretrained(model_args.model_name_or_path,
                                                  num_labels=2,
                                                  ignore_mismatched_sizes=True,
                                                  max_position_embeddings=max(512, model_args.turn_num * model_args.max_utterance_length),
                                                  use_focal_loss=model_args.loss_fct == "focal")
    elif "bert" in model_args.model_name_or_path.lower():
        model = BertForEncoderCls.from_pretrained(model_args.model_name_or_path,
                                                  num_labels=2,
                                                  max_position_embeddings=max(512, 10 * model_args.max_utterance_length),
                                                  use_focal_loss=model_args.use_focal_loss,
                                                  ignore_mismatched_sizes=True)
    else:
        raise ValueError("Unexpected model.")
    # else:
    #     model = BertForTokenClassification.from_pretrained(model_args.model_name_or_path, num_labels=2)

    model.resize_token_embeddings(len(tokenizer))

    # 训练
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                # cache_file_name=f"{data_args.train_file}".replace("txt", "arrow"),
                fn_kwargs={"tokenizer": tokenizer, "max_length": model_args.max_utterance_length},
                desc="Running tokenizer on train dataset",
            )

    # 评估
    if training_args.do_eval:
        # max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["validation"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                # cache_file_name=f"{data_args.train_file}".replace("txt", "arrow"),
                fn_kwargs={"tokenizer": tokenizer, "max_length": model_args.max_utterance_length},
                desc="Running tokenizer on validation dataset",
            )

    # 预测
    if training_args.do_predict:
        # max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["test"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                # cache_file_name=f"{data_args.train_file}".replace("txt", "arrow"),
                fn_kwargs={"tokenizer": tokenizer, "max_length": model_args.max_utterance_length},
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        predictions = np.argmax(preds, axis=-1)
        f_scores = []
        for i in range(predictions.shape[0]):
            pred = np.extract(labels[i] >= 0, predictions[i])
            label = np.extract(labels[i] >= 0, labels[i])
            equ_num = sum(pred == label)
            pred_len = sum(pred >= 0)
            label_len = sum(label >= 0)
            p = equ_num / pred_len if pred_len > 0 else 0
            r = equ_num / label_len if label_len > 0 else 0
            f = 2 * p * r / (p + r) if p + r > 0 else 0
            f_scores.append(f)
            # pred = [str(_) for _ in predictions[i].tolist()]
            # label = [str(_) for _ in labels[i].tolist() if _ >= 0]
            # pred = pred[:len(label)]
            # wd_scores.append(windowdiff("".join(pred), "".join(label), k=1))
        # print(
        #     f"pred: {predictions}\nlabel: {labels}"
        # )
        res = {"f": sum(f_scores)/len(f_scores)}
        return res

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[],
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(f"{training_args.output_dir}/best_model")  # Saves the tokenizer too for easy upload

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )
        metrics = predict_results.metrics
        logger.info(metrics)
        if trainer.is_world_process_zero():
            output_prediction_file = os.path.join(
                training_args.output_dir, f"{model_args.model_name_or_path.replace('/', '_')}_res.txt")
            predictions = predict_results.predictions[0]
            labels = predict_results.label_ids

            with open(output_prediction_file, "w", encoding="utf-8") as fw:
                for pred, label in zip(predictions, labels):
                    pred = np.argmax(pred, axis=-1)
                    fw.write(json.dumps({"pred": pred.tolist(), "label": label.tolist()}, ensure_ascii=False) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
