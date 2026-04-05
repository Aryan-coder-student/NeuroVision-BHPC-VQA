---
name: Project Setup and Advanced Training (Quantization & RLHF)
description: Outlines the workflow to bootstrap the VQA project using uv and retrain models with Quantization and RLHF (TRL).
---

# VQA Project Workflow Skill

This skill defines the methodology for environment bootstrapping and model training with Quantization and RLHF within the NeuroVision VQA app.

## Prerequisites
- A GPU capable of 4-bit quantification (Nvidia).
- `uv` package manager installed.

## 1. Environment Setup

*Bootstrap the project instantly*
// turbo
1. Run `./setup.sh` (or `bash setup.sh`) from the root directory.
   - This creates a `uv` virtual environment in `.venv`.
   - Populates necessary PIP packages including `bitsandbytes`, `peft`, and `trl`.
   - Downloads the Hugging Face `vqa-rad` dataset into `data/bronze/`.

## 2. Preprocessing Data

1. Activate your virtual environment: `source .venv/Scripts/activate`
2. Run data conversion:
   ```bash
   dvc repro preprocess
   ```
   Or run the script manually: `python src/preprocess_data.py`

## 3. Training the Model (QLoRA + RLHF)

1. Adjust `config.yaml` to point to the base model.
2. The `src/model.py` module defines the `BitsAndBytesConfig` allowing 4-bit quantized loading alongside PEFT/LoRA adapter setup.
3. Training via `src/train.py` utilizes the TRL framework (e.g., `SFTTrainer` or `DPOTrainer`) configured for supervised learning over a quantified multimodal LLM.
4. Execute training:
   ```bash
   dvc repro train
   ```
   Or manually: `python src/train.py`

## 4. Evaluation and App Serving

1. Generate answers & track metrics via BLEU score using `dvc repro evaluate`.
2. Start the API by running `python Deployment/app.py`.
3. Launch the Streamlit chat frontend.
