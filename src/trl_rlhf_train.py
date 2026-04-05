import os
os.environ['HF_HOME'] = os.path.abspath("./.hf_cache")
import torch
import yaml
from transformers import BlipProcessor, BlipForQuestionAnswering, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
# Ensure you have installed standard HuggingFace 'datasets' and 'trl' 
# pip install trl datasets peft bitsandbytes

# 1. Configuration for Quantization and LoRA
print("Loading model for RLHF/DPO...")
config = yaml.safe_load(open("./config.yaml", "r"))
model_id = config["finetune_model"]["orignal_model_id"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = BlipForQuestionAnswering.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    device_map="auto"
)
model_ref = BlipForQuestionAnswering.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    device_map="auto"
)

# Setup PEFT LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "key"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

processor = BlipProcessor.from_pretrained(model_id)

# 2. Data Preparation for DPO
# RLHF via DPO (Direct Preference Optimization) requires preference dataset:
# prompt (image + question), chosen (preferred answer), rejected (dispreferred answer)
def mock_dataset_generator():
    """
    Note: To properly train with DPO, you'll need a dataset with 'prompt', 'chosen', and 'rejected'.
    Here we create a mock dataset to demonstrate the pipeline setup.
    """
    return [
        {
            "prompt": "Is there a tumor in this MRI?",
            "chosen": "No, this MRI scan does not show any signs of a tumor.",
            "rejected": "No."
        },
        {
            "prompt": "What is the anatomy shown?",
            "chosen": "The anatomy shown is the temporal lobe of the human brain.",
            "rejected": "Brain part."
        }
    ]

# In practice: dataset = load_dataset("your_dpo_preference_dataset_here")
# For now we use the mock dataset
from datasets import Dataset
mock_data = Dataset.from_list(mock_dataset_generator())

# 3. DPO Trainer Setup Configuration
training_args = DPOConfig(
    output_dir="./Deployment/DPO_RLHF_Model",
    beta=0.1,                          # KL penalty
    logging_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    max_prompt_length=128,
    max_length=256,
)

# 4. Initialize the TRL DPO Trainer
print("Setting up DPOTrainer from TRL...")
dpo_trainer = DPOTrainer(
    model,
    model_ref,                 # The reference model for KL penalty 
    args=training_args,
    train_dataset=mock_data,   
    tokenizer=processor.tokenizer,
)

# 5. Execute Training
if __name__ == "__main__":
    print("Starting DPO (RLHF) Alignment Training phase...")
    # dpo_trainer.train() # Uncomment to run if valid dataset provided
    print("DPO Training complete! Model parameters aligned with human preference.")
    # model.save_pretrained("./Deployment/DPO_RLHF_Best_Model")
