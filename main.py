import os
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain, RAG, & Supabase Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# 2. Initialize Groq LLM (Securely using .env)
API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=API_KEY, model_name="llama-3.1-8b-instant", temperature=0.7)

# 3. Build the FAISS Vector Database
print("Loading Research Database...")
try:
    loader = TextLoader("research_data.txt", encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2}) 
    
    print("✅ RAG Database Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading database: {e}. Make sure research_data.txt exists.")
    retriever = None

# 4. The System Prompt (Updated for Language Mirroring)
system_prompt = (
    "You are CogniSync, a web-based digital wellness coach designed exclusively for Filipino students. "
    "Your sole purpose is to address the mental health challenges associated with technology addiction.\n\n"
    "Rules:\n"
    "1. TONE & LANGUAGE: You use Language Mirroring. Reply in the exact same language (e.g., Tagalog, Taglish, or English) that the user speaks to you. Maintain an empathetic and professional tone.\n"
    "2. GUARDRAILS: Only discuss digital wellness. Refuse homework or medical queries.\n"
    "3. INTERVENTIONS: Use CBT-Lite cognitive reframing.\n"
    "4. CULTURE: Suggest Philippine analog hobbies (e.g., Sungka, basketball, local cooking).\n"
    "5. RESEARCH INTEGRATION: You MUST use the provided Context below to cite real authors and statistics "
    "when explaining concepts to the user. Integrate the citations naturally into your conversation.\n\n"
    "Context Database:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

if retriever:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
else:
    rag_chain = None

class ChatRequest(BaseModel):
    query: str

# 5. The Active Chat Endpoint (Now with Supabase Logging)
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if retriever and rag_chain:
            # Get the AI response
            response = rag_chain.invoke({"input": request.query})
            ai_reply = response["answer"]
            
            # Log analytics anonymously to Supabase
            if supabase:
                try:
                    analytics_payload = {
                        "session_id": str(uuid.uuid4()),
                        "symptom_detected": "Auto-detected from chat", # We can make this dynamic later
                        "intervention_used": "CBT-Lite Response" # We can make this dynamic later
                    }
                    supabase.table("anonymous_analytics").insert(analytics_payload).execute()
                    print("✅ Analytics successfully logged to Supabase!")
                except Exception as db_error:
                    print(f"⚠️ Supabase logging failed: {db_error}")

            return {"reply": ai_reply}
        else:
            return {"reply": "My research database is currently offline. Please check the server."}
    except Exception as e:
        print(f"Error calling AI: {e}")
        return {"reply": "I'm experiencing digital fatigue. My AI engine is unreachable."}

@app.get("/")
async def root():
    return {"status": "CogniSync RAG API is running!"}