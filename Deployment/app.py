import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import BlipProcessor, BlipForQuestionAnswering
from langchain_core.tools import Tool
from langchain_community.utilities import SerpAPIWrapper, PubMedAPIWrapper
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import os
from PIL import Image
import uvicorn
import io

class Settings(BaseSettings):
    SERPAPI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    PORT: int = 7860
    CORS_ORIGINS: list[str] = ["*"]
    
    model_config = SettingsConfigDict(env_file="Deployment/.env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

# Set env vars for underlying LangChain clients that natively grab from os.environ
if settings.SERPAPI_API_KEY:
    os.environ["SERPAPI_API_KEY"] = settings.SERPAPI_API_KEY
if settings.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download model dynamically from Hugging Face Hub (uploaded via upload_model.py)
HF_REPO_ID = "pahariaryan121/NeuroVision-VQA"
fine_tuned_model = BlipForQuestionAnswering.from_pretrained(HF_REPO_ID).to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# Initialize memory using the appropriate checkpointer
memory = MemorySaver()

def chat_bot_query(query: str):
    search_tool = Tool(
        name="Medical_Web_Search",
        func=SerpAPIWrapper().run,
        description="Searches the web for medical information related to brain, CT, and MRI scans."
    )

    pubmed_tool = Tool(
        name="PubMed_Search",
        func=PubMedAPIWrapper().run,
        description="Searches PubMed for research papers related to brain, CT, and MRI scans."
    )
    
    tools = [search_tool, pubmed_tool]
    llm = ChatGroq(model="gemma2-9b-it")
    
    # Create the modern ReAct agent using langgraph
    agent_executor = create_react_agent(model=llm, tools=tools, checkpointer=memory)

    # Invoke the agent graph
    response = agent_executor.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "global_thread"}}
    )
    
    return response["messages"][-1].content


def predict_answer(image, question):
    try:
        image = image.convert('RGB')
        inputs = processor(image, question, return_tensors="pt").to(device)
        fine_tuned_output = fine_tuned_model.generate(**inputs)
        fine_tuned_answer = processor.tokenizer.decode(fine_tuned_output[0], skip_special_tokens=True)
        return fine_tuned_answer
    except Exception as e:
        return f"Error in prediction: {e}"

class ChatRequest(BaseModel):
    query: str

@app.post("/predict/")
async def predict(question: str = Form(...), file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        answer = predict_answer(image, question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        response = chat_bot_query(request.query)
        return {"response": response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=settings.PORT)