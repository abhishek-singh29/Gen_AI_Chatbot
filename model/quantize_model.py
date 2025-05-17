from transformers import BertForQuestionAnswering, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

model = BertForQuestionAnswering.from_pretrained(
    "bert-base-uncased",
    quantization_config=bnb_config,
    device_map="auto"
)

model.save_pretrained("../model/quantized_fine_tuned/")
print("✅ Quantized model saved.")
