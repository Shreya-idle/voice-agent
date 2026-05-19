import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from agent import Assistant

class TestAgent:
    def test_assistant_init(self):
        with patch('livekit.agents.Agent.__init__', return_value=None):
            assistant = Assistant()
            assert assistant is not None

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

    def test_save_voice_transcript_no_firebase(self):
        from agent import save_voice_transcript
        with patch('agent.FIREBASE_INITIALIZED', False):
            save_voice_transcript("test question", "test answer")

    def test_save_voice_transcript_success(self):
        from agent import save_voice_transcript
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection

        with patch('agent.FIREBASE_INITIALIZED', True), \
             patch('agent.db', mock_db):
            save_voice_transcript("What is your name?", "I don't know.")
            mock_db.collection.assert_called_once_with('transcripts')
            mock_collection.add.assert_called_once()
            
            added_data = mock_collection.add.call_args[0][0]
            assert added_data["uid"] == "voice_session"
            assert added_data["question"] == "What is your name?"
            assert added_data["answer"] == "I don't know."
            assert added_data["is_answered"] is False 

    def test_save_voice_transcript_success_answered(self):
        from agent import save_voice_transcript
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection

        with patch('agent.FIREBASE_INITIALIZED', True), \
             patch('agent.db', mock_db):
            save_voice_transcript("What is your name?", "Amit")
            added_data = mock_collection.add.call_args[0][0]
            assert added_data["is_answered"] is True 

    @pytest.mark.asyncio
    async def test_voice_agent_session_conversation_events(self):
        from agent import voice_agent_session
        
        mock_ctx = Mock()
        mock_ctx.room = Mock()
        
        registered_callbacks = {}
        def mock_on(event_name, callback=None):
            if callback is not None:
                registered_callbacks[event_name] = callback
                return callback
            def decorator(cb):
                registered_callbacks[event_name] = cb
                return cb
            return decorator

        mock_instance = MagicMock()
        mock_instance.on = mock_on
        mock_instance.start = AsyncMock()
        mock_instance.generate_reply = AsyncMock()
        
        with patch('agent.AgentSession', return_value=mock_instance), \
             patch('agent.save_voice_transcript') as mock_save:
            await voice_agent_session(mock_ctx)
            
            assert "conversation_item_added" in registered_callbacks
            callback = registered_callbacks["conversation_item_added"]
            
            user_event = Mock()
            user_event.item = Mock()
            user_event.item.role = "user"
            user_event.item.content = "What is Amit's GPA?"
            callback(user_event)
            
            assistant_event = Mock()
            assistant_event.item = Mock()
            assistant_event.item.role = "assistant"
            assistant_event.item.content = "Amit has a 3.9 GPA."
            callback(assistant_event)
            
            mock_save.assert_called_once_with("What is Amit's GPA?", "Amit has a 3.9 GPA.")

if __name__ == "__main__":
    pytest.main()
