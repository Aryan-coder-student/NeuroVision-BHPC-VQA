import os

# Redirect ALL caching and temp directories to the project drive to avoid C: disk pressure
project_root = os.path.dirname(os.path.dirname(__file__))
_hf_cache = os.path.join(project_root, ".hf_cache")
_tmp_dir = os.path.join(project_root, ".tmp")
os.makedirs(_hf_cache, exist_ok=True)
os.makedirs(_tmp_dir, exist_ok=True)

os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = _hf_cache
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TMPDIR"] = _tmp_dir
os.environ["TEMP"] = _tmp_dir
os.environ["TMP"] = _tmp_dir

from huggingface_hub import HfApi, create_repo, login
import getpass

def upload_model_to_hub():
    # Replace with your actual repo ID
    repo_id = "pahariaryan121/NeuroVision-VQA"
    model_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sharded-model")
    
    if not os.path.exists(model_folder):
        print(f"Error: Model folder not found at {model_folder}")
        print("Please ensure the model training has completed and weights are saved locally.")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try reading from stored token files (set by huggingface-cli login)
        for token_path in [
            os.path.join(_hf_cache, "token"),
            os.path.expanduser("~/.cache/huggingface/token"),
            os.path.join(os.environ.get("HF_HOME", ""), "token"),
        ]:
            if os.path.isfile(token_path):
                token = open(token_path).read().strip()
                print(f"🔑 Using stored token from {token_path}")
                break
    if not token:
        print("🔑 Please enter your Hugging Face Access Token with WRITE permissions (input will be hidden):")
        token = getpass.getpass("Token: ")
    
    print("\nAuthenticating...")
    try:
        login(token=token.strip(), add_to_git_credential=True)
    except Exception as e:
        print(f"❌ Login failed! Please check your token. Error: {e}")
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
        # Upload files one-by-one to avoid loading everything into RAM at once (OOM on large .safetensors)
        files = []
        for root, dirs, filenames in os.walk(model_folder):
            for fname in filenames:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, model_folder).replace("\\", "/")
                files.append((full, rel))

        print(f"Found {len(files)} files to upload.")
        for i, (full_path, rel_path) in enumerate(files, 1):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  [{i}/{len(files)}] Uploading {rel_path} ({size_mb:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {rel_path}",
            )
            print("    ✓ Done")

        print(f"\n✅ Successfully uploaded model to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        print("\nNote: Make sure you are logged in using `huggingface-cli login` and have write access.")

if __name__ == "__main__":
    upload_model_to_hub()
