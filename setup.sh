#!/bin/bash
# setup.sh - Environment Setup and Data Download for NeuroVision VQA

echo "🚀 Starting Project Setup..."

# 1. Ensure uv is installed
if ! command -v uv &> /dev/null
then
    echo "📦 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 2. Create virtual environment
echo "🌱 Creating virtual environment using uv..."
uv venv

# 3. Activate virtual environment
# Note: Cross-platform activation snippet
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"* ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# 4. Install requirements
echo "📥 Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# 5. Fetch HuggingFace Dataset
echo "📊 Downloading VQA-RAD Dataset from HuggingFace..."
mkdir -p data/bronze

python -c "
from datasets import load_dataset
import os

print('Downloading dataset flaviagiammarino/vqa-rad...')
try:
    dataset = load_dataset('flaviagiammarino/vqa-rad')
    dataset.save_to_disk('data/bronze/')
    print('✅ Dataset successfully downloaded and saved to data/bronze/')
except Exception as e:
    print(f'❌ Error downloading dataset: {e}')
"

echo "🎉 Setup Complete! You can now activate the environment using:"
echo "    source .venv/Scripts/activate (Windows)"
echo "    source .venv/bin/activate (Linux/Mac)"
