import os
os.environ['HF_HOME'] = os.path.abspath("./.hf_cache")
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml

# Initialize the BLIP model and processor
config = yaml.safe_load(open("./config.yaml", "r"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = config["finetune_model"]["orignal_model_id"]

def load_model_processor(model_path=model_id, use_quantization=True):
    print("Loading Model and Processor................")
    
    if use_quantization and torch.cuda.is_available() and device.type == "cuda":
        # Configure BitsAndBytes for 4-bit Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load base model wrapped in 4-bit
        model = BlipForQuestionAnswering.from_pretrained(
            model_path, 
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepares model for k-bit training and gradient checkpointing
        model = prepare_model_for_kbit_training(model)
        
        # Setup LoRA (Parameter Efficient Fine Tuning)
        # For Blip, common projection layers are query, value, key.
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value", "key"], 
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM" # BLIP uses a causal LM head for decoding
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("Model loaded with 4-bit quantization and LoRA adapters.")
    else:
        # Fallback to standard loading if no CUDA or quantization turned off
        model = BlipForQuestionAnswering.from_pretrained(model_path).to(device)
        print("Model loaded in full precision.")

    processor = BlipProcessor.from_pretrained(model_id)
    print(f"Model and Processor loaded successfully {model_path}  !!!")
    return model, processor
