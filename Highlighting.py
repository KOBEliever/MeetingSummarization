from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import json
import torch

highlight_path = "HighlightData.json"
HighlightData = {}
with open(highlight_path, encoding="utf-8") as f:
    HighlightData = json.load(f)

contexts = HighlightData['context']
highlights = HighlightData['highlights']

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')


def preprocess_function(contexts, highlights):
    inputs = []
    labels = []
    attention = []

    for doc_contexts, doc_highlights in zip(contexts, highlights):
        for doc_context, doc_highlight in zip(doc_contexts, doc_highlights):
            for context, highlight in zip(doc_context, doc_highlight):
                sentences = context.split('，')  # 按逗号分割句子
                sentence_highlights = [highlight[start:start + len(sentence)] for start, sentence in
                                       enumerate(sentences)]

                for i, (sentence, sentence_highlight) in enumerate(zip(sentences, sentence_highlights)):
                    encoded = tokenizer.encode_plus(
                        sentence,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=512,
                        padding='max_length'
                    )

                    # 如果句子应被突出显示，则在 [CLS] 位置标记为 1，否则标记为 0
                    cls_label = 1 if any(sentence_highlight) else 0

                    inputs.append(encoded['input_ids'])
                    labels.append(cls_label)
                    attention.append(encoded['attention_mask'])

    # 转换为 tensor
    inputs = torch.tensor(inputs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    attention = torch.tensor(attention, dtype=torch.long)
    return {'input_ids': inputs, 'attention': attention, 'labels': labels}

dataset = Dataset.from_dict(preprocess_function(contexts,highlights))

model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large", num_labels=2)

# 划分数据集
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
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


trainer.train()

# 评估模型
results = trainer.evaluate()
print(results)

# 预测
predictions = trainer.predict(test_dataset)
print(predictions)