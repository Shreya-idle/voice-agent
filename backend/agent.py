import logging
import os
import json
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent
from livekit.plugins import deepgram, openai, silero

from config import settings
from rag_engine import RAGEngine

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

try:
    if not firebase_admin._apps:
        if os.path.exists(KEY_PATH):
            logger.info(f"Found serviceAccountKey at: {KEY_PATH}")
            cred = credentials.Certificate(KEY_PATH)
            firebase_admin.initialize_app(cred)
        elif os.environ.get("FIREBASE_SERVICE_ACCOUNT"):
            logger.info("Found FIREBASE_SERVICE_ACCOUNT in environment")
            service_account_info = json.loads(os.environ.get("FIREBASE_SERVICE_ACCOUNT"))
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
        else:
            logger.info(f"NOT found serviceAccountKey at: {KEY_PATH} and FIREBASE_SERVICE_ACCOUNT not set")
            firebase_admin.initialize_app()
    db = firestore.client()
    FIREBASE_INITIALIZED = True
    logger.info("Firebase Admin initialized successfully in voice agent.")
except Exception as e:
    logger.warning(f"Firebase Admin initialization failed in voice agent: {e}")
    FIREBASE_INITIALIZED = False
    db = None

logger.info("Loading RAG knowledge base...")
rag_engine = RAGEngine(
    data_path=settings.data_path,
    persist_directory=settings.persist_directory
)

results = rag_engine.query("Tell me everything about this person's background, skills, experience, and education", k=5)
resume_context = "\n".join([r.page_content for r in results])
logger.info(f"Loaded {len(results)} context chunks from knowledge base.")


def save_voice_transcript(question: str, answer: str):
    if not FIREBASE_INITIALIZED or not db:
        logger.warning("Firebase not initialized in voice agent, cannot save transcript.")
        return
    try:
        unanswered_phrases = [
            "don't know", "do not know", "couldn't find", "could not find",
            "not mentioned", "not stated", "sorry", "no information",
            "cannot answer", "unable to answer", "out of scope", "does not say",
            "doesn't say", "not specified"
        ]
        is_answered = not any(phrase in answer.lower() for phrase in unanswered_phrases)
        
        transcript_data = {
            "uid": "voice_session",
            "question": question,
            "answer": answer,
            "is_answered": is_answered,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "metadata": {
                "model": settings.groq_model_name,
                "channel": "voice"
            }
        }
        db.collection('transcripts').add(transcript_data)
        logger.info(f"Voice transcript saved to Firestore: Q='{question}' A='{answer}'")
    except Exception as e:
        logger.error(f"Failed to save voice transcript: {e}")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are a professional and helpful voice assistant.
Your goal is to answer questions about the person described below based on their resume.
Keep your responses concise and natural for a voice conversation - under 30 words when possible.
Use casual, spoken language. Never use bullet points, markdown, or special formatting.
If the resume information does not contain the answer or you don't know it, you MUST say exactly: "I am sorry, but I don't know the answer to that based on Amit's resume."
Do NOT make up or infer answers.

Here is the resume information:
{resume_context}"""
        )


server = AgentServer()


@server.rtc_session(agent_name="voice-agent")
async def voice_agent_session(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(
            api_key=settings.deepgram_api_key,
            model="nova-2",
            language="en",
        ),
        llm=openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model=settings.groq_model_name,
        ),
        tts=deepgram.TTS(
            api_key=settings.deepgram_api_key,
        ),
        vad=silero.VAD.load(),
    )

    last_user_question = None

    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        nonlocal last_user_question
        item = event.item
        
        if hasattr(item, "role") and hasattr(item, "content"):
            role = item.role
            content = item.content
            
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join([str(block) for block in content])
            else:
                text = str(content)
                
            if not text.strip():
                return
                
            logger.info(f"Voice conversation turn: role={role}, text='{text}'")
            
            if role == "user":
                last_user_question = text
            elif role == "assistant":
                if last_user_question:
                    save_voice_transcript(last_user_question, text)
                    last_user_question = None

    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
