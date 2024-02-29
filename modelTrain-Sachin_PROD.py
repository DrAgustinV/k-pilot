# ===================== 
# SOURCE: https://sachinkhandewal.medium.com/finetuning-mistral-7b-into-a-medical-chat-doctor-using-huggingface-qlora-peft-5ce15d45f581
# GITHUB: https://github.com/sachink1729/Finetuning-Mistral-7B-Chat-Doctor-Huggingface-LoRA-PEFT/blob/main/mistral-finetuned%20(1).ipynb
# =========== JSONL VALIDATOR: https://jsonlines.org/validator/
# ========== SWAP MEMORY =============0
# https://linuxhandbook.com/increase-swap-ubuntu/

# Ran on: Sagemaker ml.g4dn.12xlarge instance, which provides you 4 Nvidia T4 GPUs (64GB VRAM)

# ===================== REQUIREMENTS
# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q -U datasets scipy ipywidgets matplotlib
# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'

import os
import torch
import bitsandbytes
import transformers
import peft
import accelerate
import datasets
import scipy
import ipywidgets
import matplotlib
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from trl import setup_chat_format


os.environ['HF_HOME'] = "huggingface_home_directory/"
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Formatting Prompts and Tokenizing the dataset.
def formatting_func(example):
    text = f"### The following is a doctor's opinion on a person's query: \n### Patient query: {example['input']} \n### Doctor opinion: {example['output']}"
    return text

# EXAMPLE of how the format looks like for train example
### The following is a doctor's opinion on a person's query: 
### Patient query: I have considerable lower back pain, also numbness in left buttocks and down left leg, girdling at the upper thigh.  MRI shows \"Small protrusiton of L3-4 interv. disc on left far laterally with annular fissuring fesulting in mild left neural foraminal narrowing with slight posterolateral displacement of the exiting l3 nerve root.\"  Other mild bulges L4-5 w/ fissuring, and mild buldge  L5-S1. 1) does this explain symptoms 2) I have a plane/car trip in 2 days lasting 8 hrs, then other travel.  Will this be harmful? 
### Doctor opinion: Hi, Your MRI report does explain your symptoms. Travelling is possible providing you take certain measures to make your journey as comfortable as possible. I suggest you ensure you take adequate supplies of your painkillers. When on the plane take every opportunity to move about the cabin to avoid sitting in the same position for too long. Likewise, when travelling by car, try to make frequent stops, so you can take a short walk to move your legs.  Chat Doctor.


# ==============================================================
# =======DATASET ======
# main dataset -> lavita/ChatDoctor-HealthCareMagic-100k
from datasets import load_dataset, load_from_disk

# https://huggingface.co/learn/nlp-course/chapter5/3
# ===== STEP 1 - load from huggingface and save to disk
# dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
# dataset.save_to_disk("/home/dragutin/projects/Colab/lchain/prep-train/Sachin/test.hf")
# ===== STEP 2 - load from disk, shuffle and select a range, save to disk
# dataset = load_from_disk("/home/dragutin/projects/Colab/lchain/prep-train/Sachin/test.hf")
# sample_sample = dataset["train"].shuffle(seed=4).select(range(100))
# sample_sample.save_to_disk("/home/dragutin/projects/Colab/lchain/prep-train/Sachin/test_100.hf")
# Peek at the first few examples
# print(sample_sample[:3])

# ===== STEP 3 - load from disk and split
dataset = load_from_disk("./test_100.hf")
# print(dataset.train_test_split(test_size=0.1))
dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']
# print(train_dataset[:3])
# print(eval_dataset[:3])

# =======================================================
# 2. Load base model
# Let's now load Mistral - mistralai/Mistral-7B-v0.1 - using 4-bit quantization!
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, resume_download=True)

# ==============================================================
# 3. Tokenization
# Now that we have initialized our tokenizer lets tokenize the dataset.
# Set up the tokenizer. Add padding on the left as it makes training use less memory.
max_length = 512 # differs from datasets to datasets
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)


# Let's get a distribution of our dataset lengths, so we can determine the appropriate max_length for our input tensors.
# import matplotlib.pyplot as plt
# def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
#     lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
#     lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
#     print(len(lengths))
#     # Plotting the histogram
#     plt.figure(figsize=(10, 6))
#     plt.hist(lengths, bins=20, alpha=0.7, color='blue')
#     plt.xlabel('Length of input_ids')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Lengths of input_ids')
#     plt.show()
# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

max_length = 512 # This was an appropriate max length for my dataset

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

print(tokenized_train_dataset[1]['input_ids'])
exit()
# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)


# How does the base model do?
query = "Hi doc i am in the recovery after dengue, sometimes my heart feels like its rubbing with my chest and it feels very uncomfortable, what can be my problem?"
eval_prompt = """Patient's Query:\n\n {} ###\n\n""".format(query)
# Re-init the tokenizer so it doesn't add padding or eos token
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True))


# ==============================================================
# 4. Set Up LoRA
# Now, to start our fine-tuning, we have to apply some preprocessing to the model to prepare it for training. 
# For that use the prepare_model_for_kbit_training method from PEFT.
from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Here we define the LoRA config.
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print_trainable_parameters(model)

# Accelerator
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model = accelerator.prepare_model(model)

# ==============================================================
# 5. Run Training!
import transformers
from datetime import datetime

project = "chat-doctor-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-4, # Want a small lr for finetuning
        #bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=25,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# ==============================================================
# ==============================================================



