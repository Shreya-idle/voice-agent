import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import os

# Set environment variables for testing
os.environ["GROQ_API_KEY"] = "test_key"
os.environ["ELEVENLABS_API_KEY"] = "test_key"

from main import app, handle_tts

client = TestClient(app)

@pytest.fixture
def mock_elevenlabs():
    with patch('main.eleven_client') as mock:
        yield mock

@pytest.fixture
def mock_firestore():
    with patch('main.db') as mock:
        yield mock

@pytest.fixture
def mock_qa_chain():
    with patch('main.agent_app.qa_chain') as mock:
        yield mock

class TestMainAPI:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Voice Agent API is running", "status": "healthy"}

    def test_analytics_no_firebase(self):
        with patch('main.FIREBASE_INITIALIZED', False):
            response = client.get("/analytics")
            assert response.status_code == 503

    def test_analytics_success(self, mock_firestore):
        mock_doc = Mock()
        mock_doc.to_dict.return_value = {"is_answered": True}
        mock_firestore.collection().stream.return_value = [mock_doc]
        
        with patch('main.FIREBASE_INITIALIZED', True):
            response = client.get("/analytics")
            assert response.status_code == 200
            data = response.json()
            assert data["total_questions"] == 1
            assert data["answered"] == 1

    def test_handle_tts_success(self, mock_elevenlabs):
        mock_gen = MagicMock()
        mock_gen.__iter__.return_value = [b"audio_chunk"]
        mock_elevenlabs.text_to_speech.convert.return_value = mock_gen
        
        with patch('os.path.join', return_value="dummy_path"), \
             patch('builtins.open', MagicMock()):
            url = handle_tts("Hello world")
            assert url is not None
            assert "/static/audio/" in url

    def test_chat_success(self, mock_qa_chain, mock_firestore):
        mock_qa_chain.invoke.return_value = {"result": "test answer", "source_documents": []}
        
        mock_user_doc = Mock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {"credits": 10}
        mock_firestore.collection().document().get.return_value = mock_user_doc
        
        with patch('main.FIREBASE_INITIALIZED', True), \
             patch('main.handle_tts', return_value="/static/audio/test.mp3"):
            response = client.post("/chat", json={"message": "hello", "uid": "user123"})
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "test answer"
            assert data["audio_url"] == "/static/audio/test.mp3"
            assert data["remaining_credits"] == 9

    def test_get_user_credits_existing(self, mock_firestore):
        mock_user_doc = Mock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {"credits": 5}
        mock_firestore.collection().document().get.return_value = mock_user_doc
        
        with patch('main.FIREBASE_INITIALIZED', True):
            response = client.get("/user/user123/credits")
            assert response.status_code == 200
            assert response.json() == {"credits": 5}

    def test_get_user_credits_new(self, mock_firestore):
        mock_user_doc = Mock()
        mock_user_doc.exists = False
        mock_firestore.collection().document().get.return_value = mock_user_doc
        
        with patch('main.FIREBASE_INITIALIZED', True):
            response = client.get("/user/user123/credits")
            assert response.status_code == 200
            assert response.json() == {"credits": 10}
