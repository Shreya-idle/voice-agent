import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import os

os.environ["GROQ_API_KEY"] = "test_key"
os.environ["ELEVENLABS_API_KEY"] = "test_key"

from main import app, handle_tts, verify_firebase_token

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_auth_dependency():
    app.dependency_overrides[verify_firebase_token] = lambda: "user123"
    if hasattr(app.state, "limiter"):
        app.state.limiter.enabled = False
    yield
    app.dependency_overrides.clear()
    if hasattr(app.state, "limiter"):
        app.state.limiter.enabled = True

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

    def test_verify_firebase_token_invalid(self):
        # Clear the autouse override so the real verify_firebase_token runs
        app.dependency_overrides.clear()
        with patch('main.FIREBASE_INITIALIZED', True), \
             patch('firebase_admin.auth.verify_id_token', side_effect=Exception("Invalid token")):
            response = client.get("/user/user123/credits", headers={"Authorization": "Bearer invalid_token"})
            assert response.status_code == 401

    def test_get_user_credits_unauthorized(self):
        # Override token verify to return userABC, but we query user123
        app.dependency_overrides[verify_firebase_token] = lambda: "userABC"
        with patch('main.FIREBASE_INITIALIZED', True):
            response = client.get("/user/user123/credits", headers={"Authorization": "Bearer token"})
            assert response.status_code == 403

    def test_get_user_credits_exception(self, mock_firestore):
        mock_firestore.collection.side_effect = Exception("DB error")
        with patch('main.FIREBASE_INITIALIZED', True):
            response = client.get("/user/user123/credits")
            assert response.status_code == 200
            assert response.json() == {"credits": 10}

    def test_chat_no_qa_chain(self):
        with patch('main.agent_app.qa_chain', None):
            response = client.post("/chat", json={"message": "hello", "uid": "user123"})
            assert response.status_code == 503

    def test_chat_out_of_credits(self, mock_firestore, mock_qa_chain):
        mock_user_doc = Mock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {"credits": 0}
        mock_firestore.collection().document().get.return_value = mock_user_doc

        with patch('main.FIREBASE_INITIALIZED', True):
            response = client.post("/chat", json={"message": "hello", "uid": "user123"})
            assert response.status_code == 200
            assert response.json()["response"] == "You have run out of credits."

    def test_chat_general_exception(self, mock_qa_chain):
        mock_qa_chain.invoke.side_effect = Exception("LLM error")
        response = client.post("/chat", json={"message": "hello", "uid": "user123"})
        assert response.status_code == 500

    def test_get_analytics_exception(self, mock_firestore):
        mock_firestore.collection.side_effect = Exception("DB error")
        with patch('main.FIREBASE_INITIALIZED', True):
            response = client.get("/analytics")
            assert response.status_code == 500

    def test_get_token_unauthorized(self):
        app.dependency_overrides[verify_firebase_token] = lambda: "userABC"
        with patch('main.FIREBASE_INITIALIZED', True):
            response = client.get("/token?room=room1&identity=user123")
            assert response.status_code == 403

    def test_get_token_missing_credentials(self):
        with patch('main.settings.livekit_api_key', None), \
             patch('main.settings.livekit_api_secret', None):
            response = client.get("/token?room=room1&identity=user123")
            assert response.status_code == 500

    def test_get_token_success(self):
        with patch('main.settings.livekit_api_key', "key"), \
             patch('main.settings.livekit_api_secret', "secret"), \
             patch('livekit.api.AccessToken.to_jwt', return_value="mocked_token"):
            response = client.get("/token?room=room1&identity=user123")
            assert response.status_code == 200
            assert response.json() == {"token": "mocked_token"}

    def test_handle_tts_no_client(self):
        with patch('main.eleven_client', None):
            assert handle_tts("Hello") is None

    def test_handle_tts_exception(self, mock_elevenlabs):
        mock_elevenlabs.text_to_speech.convert.side_effect = Exception("TTS error")
        assert handle_tts("Hello") is None
