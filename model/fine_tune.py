import json
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
with open("C:/Users/notif/Gen_AI_Assignment/data/qa_dataset.json", "r") as f:
    data = json.load(f)

# Load fast tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Function to tokenize and find start/end token positions of the answer
def encode_data(example):
    encoding = tokenizer(
        example["context"],
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    # Get answer character start/end
    answer = example["answer"]
    context = example["context"]
    start_char = context.find(answer)
    end_char = start_char + len(answer)

    # Convert character indices to token indices
    offsets = encoding["offset_mapping"][0]
    start_token = end_token = 0

    for i, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_token = i
        if start < end_char <= end:
            end_token = i
            break

    # Final dictionary to return
    encoding = {k: v.squeeze() for k, v in encoding.items() if k != "offset_mapping"}
    encoding["start_positions"] = torch.tensor(start_token)
    encoding["end_positions"] = torch.tensor(end_token)

    return encoding

# Prepare dataset for training
dataset = Dataset.from_list(data)
dataset = dataset.map(encode_data, remove_columns=["context", "question", "answer"])

# Training configuration
training_args = TrainingArguments(
    output_dir="../model/fine_tuned/",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="../logs",
    logging_strategy="epoch",
    report_to="tensorboard"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("../model/fine_tuned/")
tokenizer.save_pretrained("../model/fine_tuned/")
