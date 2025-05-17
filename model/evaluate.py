import json
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from sklearn.metrics import f1_score

model_path = "../model/fine_tuned/"
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

with open("../data/qa_dataset.json") as f:
    data = json.load(f)

preds, refs = [], []

for item in data:
    inputs = tokenizer(item["context"], item["question"], return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        )
        preds.append(answer)
        refs.append(item["answer"])

f1s = [f1_score(list(ref), list(pred), average='micro') for pred, ref in zip(preds, refs)]
avg_f1 = sum(f1s) / len(f1s)

print("Average F1 Score:", avg_f1)
