from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import torch
# 90% of the code here is copy-pasted boiler-plate code 
# from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb#scrollTo=Dv1NBUozV78l
#i only know how the important things work unsloth abstracts away everything else
#might write a script from scratch also to learn this shit but defo not anytime soon
max_seq_length = 1024#already wayyy more than enough idts anything would be >100 also lol
load_in_4bit = True
dtype = torch.bfloat16
#loading model and tokeniser
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "chuanli11/Llama-3.2-3B-Instruct-uncensored", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)
#Attaching QLoRa adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # lowkey safe choice above this OOM error prob would be high
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],#ill initally try targetting all layers instead of just query and value to see what happense
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    

    use_gradient_checkpointing = "unsloth", 
    random_state = 1738,#I'm like hey wassup hello
    use_rslora = False,  
    loftq_config = None, )
#training the model
dataset = load_dataset("parquet", data_files="updated_pairs.parquet",split="train")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    #data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer), will fuck with packing
    packing = True, # Gain in speed lwk outweighs the losses
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 12,
        warmup_steps = 5,
        num_train_epochs = 3, 
        max_steps = -1,
        learning_rate = 2e-4,
        bf16 = True,
        fp16 = False,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", 
    ),
)
print("Starting training...")
trainer_stats = trainer.train()
print(f"Training complete in {trainer_stats.metrics['train_runtime']} seconds.")
# Saving the  LoRA weights locally
model.save_pretrained("mangpt 3B")
tokenizer.save_pretrained("mangpt 3B")
#Saving 4-bit GGUF locally
model.save_pretrained_gguf("mangpt 3B", tokenizer, quantization_method = "q4_k_m")
# Testing the model to see if it works
FastLanguageModel.for_inference(model) 

test_prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a short poem about the beauty of Manware.

### Input:
You are reading the messages of a discord server. These are the 5 messages preceding this
No Preceding Messages

### Response:
'''

# Tokenize 
inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")

# Generate the response
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print("TEST OUTPUT")
print(output_text)