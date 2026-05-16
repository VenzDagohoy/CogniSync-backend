import os
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# LangChain, RAG, & Supabase Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from supabase import create_client, Client

# Load environment variables from .env file securely
load_dotenv()
app = FastAPI()

# Enable CORS for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. INITIALIZE SUPABASE (Database Logging)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# 2. INITIALIZE AI MODEL (Groq / Llama 3.1)
API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=API_KEY, model_name="llama-3.1-8b-instant", temperature=0.7)

# 3. BUILD FAISS VECTOR DATABASE (RAG)
print("Loading Research Database...")
try:
    # Load and chunk the CBT-Lite research data
    loader = TextLoader("research_data.txt", encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Convert chunks to vector embeddings using Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )

    # Store embeddings in a FAISS searchable database
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2}) 
    
    print("✅ RAG Database Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading database: {e}. Make sure research_data.txt exists.")
    retriever = None

# 4. AI SYSTEM PROMPT & CHAIN CONFIGURATION
system_prompt = (
    "You are CogniSync, a highly empathetic and secure web-based digital wellness coach designed exclusively for Filipino students. "
    "Your sole objective is to help users navigate technology addiction, screen fatigue, doomscrolling, and digital burnout.\n\n"
    
    "CRITICAL RULES & BEHAVIORS:\n"
    "1. TONE & LANGUAGE MIRRORING: You are a supportive peer, not a strict teacher. You MUST reply in the exact language or dialect "
    "the user speaks to you (e.g., pure English, Taglish, deep Tagalog, or Bisaya/Cebuano). Seamlessly adapt to their slang and conversational energy.\n"
    
    "2. STRICT GUARDRAILS (The Pivot Strategy): You are strictly for digital wellness. "
    "- If a user asks you to do their homework, write code, or summarize modules, gently refuse and pivot by asking if the academic workload is causing screen fatigue. "
    "- If a user exhibits signs of severe clinical depression or asks for medical diagnoses, clarify that you are a digital wellness AI, not a doctor, and gently encourage them to speak to their university guidance counselor.\n"
    
    "3. CBT-LITE FRAMEWORK: Do not just tell the user to stop scrolling. Instead, use this 3-step cognitive reframing process: "
    "a) VALIDATE their feelings first (e.g., 'It makes sense you are mindlessly scrolling after a long day of classes'). "
    "b) REFRAME the thought (e.g., shifting from 'I am lazy' to 'My brain is overstimulated'). "
    "c) SUGGEST a micro-action to break the cycle.\n"
    
    "4. FILIPINO CULTURAL NUANCE: Ground your advice in the reality of the Philippine student experience (e.g., slow internet frustrations, "
    "'puyat' culture, academic pressure for board exams). Suggest culturally relevant offline 'analog' breaks, such as "
    "playing Sungka, helping cook adobo or sinigang, sweeping the yard, playing basketball at the local court, or simply resting outside.\n"
    
    "5. RESEARCH INTEGRATION (RAG): You will be provided with verified research in the Context Database below. "
    "You MUST use this data to inform your advice. When citing concepts, authors, or statistics from the database, do not sound like a robot reading a textbook. "
    "Weave the citations naturally into your conversational empathy.\n\n"
    
    "Context Database:\n{context}"
)

# Update prompt to accept session memory (chat_history)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), # short-term memory 
    ("human", "{input}"),
])

if retriever:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
else:
    rag_chain = None

# 5. DATA MODELS FOR SESSION MEMORY
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] 


# 6. ACTIVE CHAT ENDPOINT
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if not retriever or not rag_chain:
            return {"reply": "My research database is currently offline. Please check the server."}
        
        if not request.messages:
            return {"reply": "No messages received."}

        # Separate the chat history from the user's latest message
        latest_query = request.messages[-1].content
        history_messages = request.messages[:-1]
        
        # Convert history into LangChain message objects
        chat_history = []
        for msg in history_messages:
            if msg.role == "user" or msg.role == "human":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant" or msg.role == "ai":
                chat_history.append(AIMessage(content=msg.content))

        # Invoke the AI with the latest query AND the session memory
        response = rag_chain.invoke({
            "input": latest_query,
            "chat_history": chat_history
        })
        ai_reply = response["answer"]
        
        # Log analytics anonymously to Supabase
        if supabase:
            try:
                analytics_payload = {
                    "session_id": str(uuid.uuid4()),
                    "symptom_detected": "Auto-detected from chat",
                    "intervention_used": "CBT-Lite Response"
                }
                supabase.table("anonymous_analytics").insert(analytics_payload).execute()
                print("✅ Analytics successfully logged to Supabase!")
            except Exception as db_error:
                print(f"⚠️ Supabase logging failed: {db_error}")

        return {"reply": ai_reply}

    except Exception as e:
        print(f"Error calling AI: {e}")
        return {"reply": "I'm experiencing digital fatigue. My AI engine is unreachable."}

@app.get("/")
async def root():
    return {"status": "CogniSync RAG API is running securely!"}
