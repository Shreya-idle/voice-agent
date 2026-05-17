import logging
import os
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
import uuid

import firebase_admin
from firebase_admin import credentials, firestore

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from elevenlabs.client import ElevenLabs
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from livekit import api
from rag_engine import RAGEngine
from config import settings
from schemas import ChatRequest, ChatResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

eleven_client = ElevenLabs(api_key=settings.elevenlabs_api_key) if settings.elevenlabs_api_key else None
print(f"DEBUG: ElevenLabs API Key: {settings.elevenlabs_api_key[:5]}..." if settings.elevenlabs_api_key else "DEBUG: ElevenLabs API Key NOT FOUND")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

try:
    if not firebase_admin._apps:
        if os.path.exists(KEY_PATH):
            print(f"✅ Found serviceAccountKey at: {KEY_PATH}")
            cred = credentials.Certificate(KEY_PATH)
            firebase_admin.initialize_app(cred)
        else:
            print(f"❌ NOT found at: {KEY_PATH}")
            firebase_admin.initialize_app()
    db = firestore.client()
    FIREBASE_INITIALIZED = True
    logger.info("Firebase Admin initialized successfully.")
except Exception as e:
    logger.warning(f"Firebase Admin initialization failed: {e}")
    FIREBASE_INITIALIZED = False
    db = None

class VoiceAgentApp:
    def __init__(self):
        self.rag_engine: Optional[RAGEngine] = None
        self.qa_chain: Optional[RetrievalQA] = None

    def initialize(self):
        logger.info("Initializing Voice Agent components...")
        try:
            self.rag_engine = RAGEngine(
                data_path=settings.data_path,
                persist_directory=settings.persist_directory
            )
            
            llm = ChatGroq(
                temperature=settings.temperature,
                groq_api_key=settings.groq_api_key,
                model_name=settings.groq_model_name
            )

            template = """
            You are a helpful and professional Voice Agent. Use the following pieces of context to answer the user's question. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep your responses concise and suitable for voice synthesis.

            Context: {context}
            Question: {question}

            Helpful Answer:"""
            
            prompt = PromptTemplate.from_template(template)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.rag_engine.vector_store.as_retriever(),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
                
            )
            logger.info("Initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize Voice Agent: {e}")
            raise

agent_app = VoiceAgentApp()

@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_app.initialize()
    yield

app = FastAPI(
    title="Voice Agent API",
    description="Backend API for the RAG-powered Voice Agent",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "Voice Agent API is running", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
   
    if not agent_app.qa_chain:
        raise HTTPException(
            status_code=503, 
            detail="Voice Agent components are not initialized"
        )
    
    remaining_credits = 10 
    if FIREBASE_INITIALIZED and db and request.uid:
        user_ref = db.collection('users').document(request.uid)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            user_ref.set({"credits": 10})
        else:
            remaining_credits = user_doc.to_dict().get("credits", 0)
            
        if remaining_credits <= 0:
            return ChatResponse(
                response="You have run out of credits.",
                audio_url=None,
                remaining_credits=0
            )

    try:
        logger.info(f"Processing query: {request.message}")
        
        result = agent_app.qa_chain.invoke({"query": request.message})
        answer = result.get("result", "I'm sorry, I couldn't find an answer to that.")
        source_docs = result.get("source_documents", [])
        print(f"Sources used: {len(source_docs)}")
        for doc in source_docs:
            print(f"   - {doc.metadata.get('source', 'unknown')}: {doc.page_content[:100]}")
    
        audio_url = handle_tts(answer)

        if FIREBASE_INITIALIZED and db:
            unanswered_phrases = ["don't know", "couldn't find an answer", "i'm sorry"]
            is_answered = not any(phrase in answer.lower() for phrase in unanswered_phrases)
            
            try:
                transcript_data = {
                    "uid": request.uid,
                    "question": request.message,
                    "answer": answer,
                    "is_answered": is_answered,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "metadata": {
                        "model": settings.groq_model_name,
                        "remaining_credits": remaining_credits - 1 if request.uid else remaining_credits
                    }
                }
                db.collection('transcripts').add(transcript_data)
                logger.info(f"Transcript saved for user: {request.uid}")
            except Exception as e:
                logger.error(f"Failed to save transcript: {e}")

        if FIREBASE_INITIALIZED and db and request.uid:
            remaining_credits -= 1
            user_ref.update({"credits": remaining_credits})

        return ChatResponse(response=answer, audio_url=audio_url, remaining_credits=remaining_credits)

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing your request: {str(e)}"
        )

@app.get("/analytics")
async def get_analytics():
    if not FIREBASE_INITIALIZED or not db:
        raise HTTPException(status_code=503, detail="Analytics unavailable (Firebase not initialized)")
    
    try:
        transcripts = db.collection('transcripts').stream()
        total = 0
        answered = 0
        unanswered = 0
        
        for doc in transcripts:
            data = doc.to_dict()
            total += 1
            if data.get('is_answered', True):
                answered += 1
            else:
                unanswered += 1
                
        return {
            "total_questions": total,
            "answered": answered,
            "unanswered": unanswered,
            "success_rate": (answered / total * 100) if total > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/token")
async def get_token(room: str, identity: str):
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials are not configured"
        )
    
    try:
        token = api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret) \
            .with_identity(identity) \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room,
            ))
        
        return {"token": token.to_jwt()}
    except Exception as e:
        logger.error(f"Error generating token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def handle_tts(text: str) -> Optional[str]:
    logger.info(f"Generating TTS for text: {text[:50]}...")
    if not eleven_client:
        logger.warning("ElevenLabs client not initialized. Check API key.")
        return None
    
    try:
        audio_generator = eleven_client.text_to_speech.convert(
            text=text,
            voice_id="EXAVITQu4vr4xnSDxMaL",  # Bella's ID (Works on Free Tier)
            model_id="eleven_multilingual_v2"
        )
        
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join("static", "audio", filename)
        
        logger.info(f"Saving audio to {filepath}")
        audio_bytes = b"".join(list(audio_generator))
        
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
            
        url = f"/static/audio/{filename}"
        logger.info(f"TTS generated successfully: {url}")
        return url
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.host, 
        port=settings.port, 
        reload=True
    )
