import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
checkpoint_path = "outputs/checkpoint-1650"  #latest checkpoint

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="cuda"
)

model = PeftModel.from_pretrained(base_model, checkpoint_path)

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a poem about the beauty of manware
### Input:
You are a helpful AI assistant that talks in the slang and style of this particular discord server. While using the slang of the server never forget that your main goal is to *answer the question in a manner releavnt to what's asked/said* in a humourous manner.These are the 5 messages preceding this: No Preceding Messages

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens=128, 
    temperature=1.0, 
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print("Output: ")

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
