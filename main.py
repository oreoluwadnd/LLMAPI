from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableWithMessageHistory
import os
from langchain.schema import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from typing import List, Tuple
import torch
import uuid
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForMaskedLM
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.schema import StrOutputParser

app = FastAPI()

MODEL_CONFIGS = {
    "distilgpt2": "distilgpt2",
    "distilbert": "distilbert-base-uncased",
    "bert-tiny": "prajjwal1/bert-tiny",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

script_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(script_dir, "pretrained_models")
# Dictionary to store conversations for each model
conversations = {}

class Query(BaseModel):
    model: str
    question: str

def get_chat_history(messages: List[Tuple[str, str]]) -> str:
    buffer = []
    for human, ai in messages:
        buffer.append(f"Human: {human}")
        buffer.append(f"AI: {ai}")
    return "\n".join(buffer)

def create_conversation_chain(local_llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = (
        RunnablePassthrough.assign(
            history=lambda x: x["history"].buffer_as_messages if x.get("history") else []
        )
        | prompt
        | local_llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: ConversationBufferMemory(return_messages=True),
        input_messages_key="input",
        history_messages_key="history",
    )

@app.post("/query")
async def process_query(query: Query):
    model_name = query.model.lower()
    
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail="Invalid model selection")
    
    if model_name not in conversations:
        model_id = MODEL_CONFIGS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=" ")
        
        if "bert-tiny" in model_name:
            model = AutoModelForMaskedLM.from_pretrained(model_id, token=" ")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, token=" ")
        
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512 if "bert" in model_name else 1000,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.15,
            device=device,
            truncation=True
        )
        
        local_llm = HuggingFacePipeline(pipeline=pipe)
        conversations[model_name] = create_conversation_chain(local_llm)
    session_id = str(uuid.uuid4())
    
    response = conversations[model_name].invoke(
        {"input": query.question},
        {"configurable": {"session_id": session_id}}
    )
    return {"model": model_name, "response": response}