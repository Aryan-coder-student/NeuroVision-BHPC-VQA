import os
from transformers import BlipForQuestionAnswering

def shard_model():
    project_root = os.path.dirname(os.path.dirname(__file__))
    input_model_dir = os.path.join(project_root, "models", "last-saved-model")
    output_model_dir = os.path.join(project_root, "models", "sharded-model")
    
    print(f"Loading model from {input_model_dir}...")
    # Load the model strictly preferring safetensors
    model = BlipForQuestionAnswering.from_pretrained(
        input_model_dir, 
        use_safetensors=True, 
        device_map="cpu"
    )
    
    print(f"Saving sharded model to {output_model_dir} (max chunk size: 400MB)...")
    os.makedirs(output_model_dir, exist_ok=True)
    
    # max_shard_size forces it to break the safetensors into multiple 400MB files
    model.save_pretrained(
        output_model_dir, 
        max_shard_size="400MB",
        safe_serialization=True
    )
    print("Done! The model is now sharded and ready for upload.")

if __name__ == "__main__":
    shard_model()
