import os
from huggingface_hub import HfApi, create_repo

def upload_model_to_hub():
    # Replace with your actual repo ID
    repo_id = "pahariaryan121/NeuroVision-VQA"
    model_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "last-saved-model")
    
    if not os.path.exists(model_folder):
        print(f"Error: Model folder not found at {model_folder}")
        print("Please ensure the model training has completed and weights are saved locally.")
        return

    print(f"Connecting to Hugging Face Hub to upload {model_folder} to {repo_id}...")
    api = HfApi()
    
    try:
        # Create repository if it doesn't exist
        create_repo(repo_id, exist_ok=True, private=False)
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Warning/Error creating repo: {e}")

    try:
        api.upload_folder(
            folder_path=model_folder,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload fine-tuned VQA model via deployment script"
        )
        print(f"✅ Successfully uploaded model to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        print("\nNote: Make sure you are logged in using `huggingface-cli login` and have write access.")

if __name__ == "__main__":
    upload_model_to_hub()
