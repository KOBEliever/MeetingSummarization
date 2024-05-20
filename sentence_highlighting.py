from datasets import Dataset
from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments
import json
import torch

"""
===========================
数据读取
===========================
"""

lines=[]
with open('VCSum/vcsum_data/overall_context.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
data_context = [json.loads(line.rstrip('\n')) for line in lines]

lines=[]
with open('VCSum/vcsum_data/overall_highlights.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
data_highlight = [json.loads(line.rstrip('\n')) for line in lines]

context_list = [d["context"] for d in data_context]
highlight_list = [d["highlights"] for d in data_highlight]

"""
===========================
将数据处理为模型需要的格式
===========================
"""

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="google-bert/bert-base-chinese",
    cache_dir=None,
    force_download=False,)

# 将context_list和highlight_list组合成数据字典列表
data_dicts = []
for context_group, highlights_group in zip(context_list, highlight_list):
    for ctx, hls in zip(context_group, highlights_group):
        # 将发言列表转换为单个字符串
        context_str = " ".join(ctx)
        # 对每个发言进行分词
        inputs = tokenizer(context_str, padding=True, truncation=True, return_tensors="pt")
        # 添加数据点
        data_dicts.append({
            "input_ids": inputs["input_ids"].squeeze(0).tolist(),  # 转换为列表
            "attention_mask": inputs["attention_mask"].squeeze(0).tolist(),
            "highlights": hls
        })

# 使用from_list()创建Dataset对象
dataset = Dataset.from_list(data_dicts)
# 划分数据集
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()
"""
===========================
模型调用及训练
===========================
"""
model = BertForTokenClassification.from_pretrained("google-bert/bert-base-chinese", num_labels=2)
model.cuda()
# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# 训练模型
trainer.train()

# 评估模型
trainer.evaluate()