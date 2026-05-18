import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from agent import Assistant

class TestAgent:
    def test_assistant_init(self):
        with patch('livekit.agents.Agent.__init__', return_value=None):
            assistant = Assistant()
            assert assistant is not None
            # The instructions are dynamic and contain the resume context
            # We just check if it's initialized

    @pytest.mark.asyncio
    async def test_voice_agent_session(self):
        from agent import voice_agent_session
        
        mock_ctx = Mock()
        mock_ctx.room = Mock()
        
        mock_instance = MagicMock()
        mock_instance.on = MagicMock(return_value=lambda x: x)
        mock_instance.start = AsyncMock()
        mock_instance.generate_reply = AsyncMock()
        
        with patch('agent.AgentSession', return_value=mock_instance) as mock_session:
            await voice_agent_session(mock_ctx)
            mock_session.assert_called_once()
            mock_instance.start.assert_called_once()
            mock_instance.generate_reply.assert_called_once()

if __name__ == "__main__":
    pytest.main()
