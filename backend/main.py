import logging
import os
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
import uuid

import firebase_admin
from firebase_admin import credentials, firestore, auth

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
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
            print(f"Found serviceAccountKey at: {KEY_PATH}")
            cred = credentials.Certificate(KEY_PATH)
            firebase_admin.initialize_app(cred)
        elif os.environ.get("FIREBASE_SERVICE_ACCOUNT"):
            print("Found FIREBASE_SERVICE_ACCOUNT in environment")
            service_account_info = json.loads(os.environ.get("FIREBASE_SERVICE_ACCOUNT"))
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
        else:
            print(f"NOT found at: {KEY_PATH} and FIREBASE_SERVICE_ACCOUNT not set")
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
            If the context does not contain the answer or you don't know the answer, you MUST say exactly: "I am sorry, but I don't know the answer."
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

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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
@limiter.limit("20/minute")
async def root(request: Request):
    return {"message": "Voice Agent API is running", "status": "healthy"}

security = HTTPBearer()

def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not FIREBASE_INITIALIZED:
        return "unauthenticated_uid" 
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token['uid']
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/user/{uid}/credits")
@limiter.limit("10/minute")
async def get_user_credits(uid: str, request: Request, auth_uid: str = Depends(verify_firebase_token)):
    if auth_uid != uid and FIREBASE_INITIALIZED:
        raise HTTPException(status_code=403, detail="Not authorized to access these credits")
    if not FIREBASE_INITIALIZED or not db:
        return {"credits": 10}
    try:
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()
        if not user_doc.exists:
            user_ref.set({"credits": 10})
            return {"credits": 10}
        return {"credits": user_doc.to_dict().get("credits", 10)}
    except Exception as e:
        logger.error(f"Failed to fetch credits: {e}")
        return {"credits": 10}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, chat_request: ChatRequest, auth_uid: str = Depends(verify_firebase_token)):
    user_ref = None
    remaining_credits = 10
   
    effective_uid = auth_uid if FIREBASE_INITIALIZED else chat_request.uid
   
    if not agent_app.qa_chain:
        raise HTTPException(
            status_code=503, 
            detail="Voice Agent components are not initialized"
        )
    
    if FIREBASE_INITIALIZED and db and effective_uid:
        user_ref = db.collection('users').document(effective_uid)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            user_ref.set({"credits": 10})
            remaining_credits = 10
        else:
            remaining_credits = user_doc.to_dict().get("credits", 0)
            
        if remaining_credits <= 0:
            return ChatResponse(
                response="You have run out of credits.",
                audio_url=None,
                remaining_credits=0
            )

    try:
        logger.info(f"Processing query: {chat_request.message}")
        
        result = agent_app.qa_chain.invoke({"query": chat_request.message})
        answer = result.get("result", "I'm sorry, I couldn't find an answer to that.")
        source_docs = result.get("source_documents", [])
        print(f"Sources used: {len(source_docs)}")
        for doc in source_docs:
            print(f"   - {doc.metadata.get('source', 'unknown')}: {doc.page_content[:100]}")
    
        audio_url = handle_tts(answer)

        if FIREBASE_INITIALIZED and db:
            unanswered_phrases = [
                "don't know", "do not know", "couldn't find", "could not find",
                "not mentioned", "not stated", "sorry", "no information",
                "cannot answer", "unable to answer", "out of scope", "does not say",
                "doesn't say", "not specified"
            ]
            is_answered = not any(phrase in answer.lower() for phrase in unanswered_phrases)
            
            try:
                transcript_data = {
                    "uid": effective_uid,
                    "question": chat_request.message,
                    "answer": answer,
                    "is_answered": is_answered,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "metadata": {
                        "model": settings.groq_model_name,
                        "remaining_credits": remaining_credits - 1 if effective_uid else remaining_credits
                    }
                }
                db.collection('transcripts').add(transcript_data)
                logger.info(f"Transcript saved for user: {effective_uid}")
            except Exception as e:
                logger.error(f"Failed to save transcript: {e}")

        if FIREBASE_INITIALIZED and db and effective_uid:
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
@limiter.limit("10/minute")
async def get_analytics(request: Request):
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
@limiter.limit("5/minute")
async def get_token(room: str, identity: str, request: Request, auth_uid: str = Depends(verify_firebase_token)):
    if auth_uid != identity and FIREBASE_INITIALIZED:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to request token for this identity"
        )
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
            voice_id="EXAVITQu4vr4xnSDxMaL",
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
