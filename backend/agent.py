import logging
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent
from livekit.plugins import deepgram, openai, silero

from config import settings
from rag_engine import RAGEngine

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

logger.info("Loading RAG knowledge base...")
rag_engine = RAGEngine(
    data_path=settings.data_path,
    persist_directory=settings.persist_directory
)

results = rag_engine.query("Tell me everything about this person's background, skills, experience, and education", k=5)
resume_context = "\n".join([r.page_content for r in results])
logger.info(f"Loaded {len(results)} context chunks from knowledge base.")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are a professional and helpful voice assistant.
Your goal is to answer questions about the person described below based on their resume.
Keep your responses concise and natural for a voice conversation — under 30 words when possible.
Use casual, spoken language. Never use bullet points, markdown, or special formatting.
If you don't know the answer, politely say so.
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
            model="nova-3",
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

    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
