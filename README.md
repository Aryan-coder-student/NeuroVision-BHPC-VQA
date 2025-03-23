# Visual Question Answering (VQA) System

<div align="center">
  <img src="https://img.shields.io/badge/Framework-BLIP-blue" alt="BLIP Framework" />
  <img src="https://img.shields.io/badge/Built%20with-PyTorch-orange" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Pipeline-DVC-green" alt="DVC Pipeline" />
  <img src="https://img.shields.io/badge/Deployment-Docker-lightblue" alt="Docker" />
  <img src="https://img.shields.io/badge/Agent-LangChain-purple" alt="LangChain" />
</div>
<div style="display:justify-content;">
  <img src="https://github.com/user-attachments/assets/2d6bf915-cecf-4482-bde1-859eaf5fa399">
</div>


## 📑 Table of Contents
- [Dataset Download](#-dataset-download)
- [Project Structure](#-project-structure)
- [Core Components](#-core-components)
- [API Architecture](#-api-architecture)
- [LangChain Agent System](#-langchain-agent-system)
- [Docker Setup](#-docker-setup)
- [API Documentation](#-api-documentation)
- [DVC Pipeline](#-dvc-pipeline)
- [Installation](#-installation)
- [Dataset Download](#-dataset-download)
- [Results & Future Improvements](#-results--future-improvements)

---

## 📥 Dataset Download  
The VQA-RAD dataset is available on Hugging Face: [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad).  
To download the **VQA-RAD** dataset from Hugging Face and store it in `data/bronze/`, run the following script:  

```python
from datasets import load_dataset

dataset = load_dataset("flaviagiammarino/vqa-rad")
dataset.save_to_disk("data/bronze/")
```

This ensures that your dataset is properly downloaded and stored in the `bronze` data layer. 🚀

This VQA model enables machines to understand and answer natural language questions about images. Leveraging the BLIP framework, it achieves high accuracy in interpreting visual content and generating relevant textual responses. The system is enhanced with a LangChain-powered agent for handling medical queries beyond image analysis.

## 📁 Project Structure

```
VQA/
│── .dvc/                 # DVC-related files for data versioning
│── config/               # Configuration files
│── dags/                 # DAG workflows for pipeline execution
│── data/                 # Dataset storage
│   ├── bronze/           # Raw data
│   ├── silver/           # Processed data
│   ├── .gitignore        # Ignore large files
│   ├── bronze.dvc        # DVC tracking file for raw data
│── Deployment/           # Deployment-related files
│   ├── __pycache__/      # Python cache files
│   ├── Model_14...       # Saved fine-tuned model
│   ├── streamlit/        # Streamlit UI
│   │   ├── .gitignore    # Ignore unnecessary files in Streamlit
│   │   ├── .env          # Environment variables
│   │   ├── main.py       # Streamlit app entry point
│   ├── app.py            # Flask API entry point
│   ├── test_api.py       # API testing script
│── logs/                 # Logs for debugging and tracking
│── mlruns/               # MLflow experiment tracking
│── models/               # Directory for storing trained models
│── notebooks/            # Jupyter notebooks for model exploration
│── plugins/              # Additional utilities or extensions
│── results/              # Evaluation results storage
│── src/                  # Core source code
│   ├── __pycache__/      # Python cache files
│   ├── test/             # Test scripts
│   ├── model.py          # Model loading and inference
│   ├── preprocess_data.py # Data preprocessing script
│   ├── train.py          # Model training script
│   ├── evaluate.py       # Model evaluation script
│── .dvcignore            # Ignore files for DVC
│── .gitignore            # Ignore unnecessary files for Git
│── config.yaml           # Main configuration file
│── docker-compose.yml    # Docker Compose setup
│── Dockerfile            # Docker container setup
│── dvc.yaml              # DVC pipeline configuration
│── dvc.lock              # DVC lock file
│── params.yaml           # Hyperparameter settings
│── README.md             # Project documentation
│── requirements.txt      # Python dependencies
```

## 🧩 Core Components

### `model.py`

- ⚙️ Loads the **BLIP model for Visual Question Answering**
- 🔄 Initializes the model and processor
- ⚡ Fetches model configurations from `config.yaml`
- 📤 Returns both the model and its processor for inference and training

### `train.py`

- 🏋️ Handles **model training** using **PyTorch**
- 📊 Loads preprocessed datasets
- 🚀 Implements **gradient accumulation** and **mixed precision training**
- 📈 Tracks **training loss, validation loss, and BLEU score**
- 📝 Uses **MLflow** for experiment tracking
- 🛑 Implements **early stopping** to save the best model
- 💾 Saves the trained model in `Deployment/Model/`

### `evaluate.py`

- ⚖️ Loads the trained model and the test dataset
- 🎯 Generates answers using the model
- 📏 Computes **BLEU scores** to measure model performance
- 📁 Saves evaluation results as **JSON files** in `results/`
- 🔍 Compares the last saved model with the best-performing model

### `preprocess_data.py`

- 📥 Loads the **VQA dataset**
- 🖼️ Converts images to RGB format and tokenizes text data
- 🔄 Uses the BLIP processor to encode images and questions
- 📤 Saves the processed data as **pickle files** in `data/silver/`

## 🌐 API Architecture

### `app.py` - Flask API Implementation

The Flask API serves as the backbone of our deployment strategy, offering two primary endpoints:

1. **Image Question Answering (`/predict/`)**
   - Receives an image file and a question
   - Processes the image using the BLIP processor
   - Passes the processed inputs to the fine-tuned model
   - Returns the generated answer as a JSON response

2. **Medical Chatbot (`/chat/`)**
   - Accepts a text query related to medical topics
   - Forwards the query to a LangChain-powered agent
   - Returns comprehensive medical information from multiple sources

**Key Components:**
- **CORS Support**: Enables cross-origin requests for frontend integration
- **Environment Variables**: Securely loads API keys for external services
- **GPU/CPU Detection**: Automatically selects the appropriate device for model inference
- **Error Handling**: Implements robust exception handling for all endpoints

## 🤖 LangChain Agent System

Our system employs a sophisticated LangChain agent architecture to handle medical queries beyond image analysis:

### Agent Components

1. **Search Tools**
   - **Medical Web Search**: Utilizes SerpAPI to search the web for medical information related to brain, CT, and MRI scans
   - **PubMed Search**: Connects to PubMed API for accessing peer-reviewed medical research papers

2. **LLM Backend**
   - Powered by Groq's **Gemma2-9b-it** model for generating coherent and accurate responses
   - Configured for medical domain specialization

3. **Memory System**
   - Implements `ConversationBufferMemory` to maintain context across multiple queries
   - Enables follow-up questions and contextual understanding

4. **Agent Configuration**
   - Uses `ZERO_SHOT_REACT_DESCRIPTION` agent type for reasoning capabilities
   - Limited to 10 iterations to ensure timely responses
   - Includes error handling for parsing issues

5. **Flow Process**
   - Receives user query via the `/chat/` endpoint
   - Agent analyzes the query and determines which tools to use
   - Searches appropriate sources (web or PubMed)
   - Synthesizes information into a comprehensive response
   - Returns formatted answer to the user

## 🐳 Docker Setup

### `Dockerfile`

The `Dockerfile` defines the containerized environment for deploying the VQA model. It installs all dependencies, copies source files, and sets up both the **Flask API** and **Streamlit UI**.

#### **Dockerfile Breakdown:**

```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000 8501
CMD ["sh", "-c", "python Deployment/app.py"]
```

#### **Build and Run Commands:**

```bash
# Build the Docker image
docker build -t vqa-app .

# Run the container
docker run -p 5000:5000 -p 8501:8501 vqa-app
```

## 📡 API Documentation

### Predict Answer Endpoint

```bash
# Request format
curl -X POST "http://127.0.0.1:5000/predict/" \
  -F "file=@path/to/image.jpg" \
  -F "question=What is in the image?"
```

**Response:**
```json
{
  "answer": "The image shows a brain MRI scan with a visible tumor in the temporal lobe"
}
```

### Chatbot Query Endpoint

```bash
# Request format
curl -X POST "http://127.0.0.1:5000/chat/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Astrocytoma?"}'
```

**Response:**
```json
{
  "response": "Astrocytoma is a type of brain tumor that develops from star-shaped cells called astrocytes. These cells are part of the glial tissue, which supports and protects neurons in the brain and spinal cord. Astrocytomas can range from slow-growing (low-grade) to aggressive (high-grade) tumors. They are classified by the World Health Organization (WHO) into four grades (I-IV), with grade IV being the most aggressive form, also known as glioblastoma multiforme. Symptoms may include headaches, seizures, memory problems, and changes in behavior, depending on the tumor's location and size. Treatment typically involves surgery, radiation therapy, and chemotherapy, depending on the grade and location of the tumor."
}
```

### API Testing (`test_api.py`)

```python
import requests

# Test the chat endpoint
url = "http://127.0.0.1:5000/chat/"
payload = {"query": "What is Astrocytoma?"}
response = requests.post(url, json=payload)
print(response.json())
```

## 📦 DVC Pipeline

### **Data Version Control Pipeline**

DVC manages data pipelines efficiently, tracking data, models, and experiments while ensuring reproducibility.

#### **Pipeline Stages:**

1. **Data Preprocessing Stage** 🔄
   - **Input:** Raw data from `data/bronze/`
   - **Process:** `preprocess_data.py` cleans and prepares the dataset
   - **Output:** Processed data stored in `data/silver/`

2. **Model Training Stage** 🏋️
   - **Input:** Preprocessed data from `data/silver/` and hyperparameters from `params.yaml`
   - **Process:** `train.py` trains the VQA model using PyTorch
   - **Output:** Trained model saved in `Deployment/Model/`

3. **Model Evaluation Stage** ⚖️
   - **Input:** Trained model from `Deployment/Model/` and test dataset from `data/silver/`
   - **Process:** `evaluate.py` evaluates model performance using BLEU scores
   - **Output:** Evaluation results saved in `results/`

### **DVC Pipeline Configuration (`dvc.yaml`)**

```yaml
stages:
  preprocess:
    cmd: python src/preprocess_data.py
    deps:
      - src/preprocess_data.py
      - data/bronze/
    outs:
      - data/silver/
  
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/silver/
      - params.yaml
    outs:
      - Deployment/Model/
  
  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - Deployment/Model/
      - data/silver/
    outs:
      - results/
```

### **DVC Commands**

```bash
# Execute all pipeline stages in sequence
dvc repro

# Push data and models to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

## 📌 Installation

Install all required dependencies with:

```bash
pip install -r requirements.txt
```

## 📊 Results & Future Improvements

### Current Results

- 📈 Logs comprehensive metrics via MLflow
- 📁 Stores detailed evaluation results in `./results/`
- 💾 Saves the fine-tuned model in `Deployment/Model/`

### ✨ Future Work

- 🔬 Experiment with different model architectures
- 📊 Add visualization tools for better model interpretation
- 🌐 Expand language support for multilingual VQA
- 🔄 Implement continuous learning capabilities
- 🧠 Enhance the LangChain agent with more specialized medical tools
- 🔍 Add image segmentation capabilities for more detailed medical image analysis

---

<div align="center">
  <p><b>Visual Question Answering Model</b> | Powered by BLIP Framework</p>
</div>
<div align="center">
  <img src="https://img.shields.io/badge/Framework-BLIP-blue" alt="BLIP Framework" />
  <img src="https://img.shields.io/badge/Built%20with-PyTorch-orange" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Agent-LangChain-purple" alt="LangChain" />
</div>
