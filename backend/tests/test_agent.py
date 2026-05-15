import pytest
from unittest.mock import Mock, patch, AsyncMock
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
        
        with patch('agent.AgentSession', return_value=AsyncMock()) as mock_session:
            await voice_agent_session(mock_ctx)
            mock_session.assert_called_once()
            # Verify session start was called
            mock_session.return_value.start.assert_called_once()
            # Verify generate_reply was called
            mock_session.return_value.generate_reply.assert_called_once()

if __name__ == "__main__":
    pytest.main()
