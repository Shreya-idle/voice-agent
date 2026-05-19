import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app, verify_firebase_token

@pytest.fixture(autouse=True)
def clean_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()

def test_unauthenticated_requests_blocked():
    client = TestClient(app)
    
    resp1 = client.get("/user/some_uid/credits")
    assert resp1.status_code == 401
    
    resp2 = client.post("/chat", json={"message": "hello", "uid": "some_uid"})
    assert resp2.status_code == 401

    resp3 = client.get("/token?room=test&identity=some_uid")
    assert resp3.status_code == 401

def test_idor_blocked_when_authenticated():
    client = TestClient(app)
    
    app.dependency_overrides[verify_firebase_token] = lambda: "legitimate_user"
    
    response = client.get("/user/victim_user/credits", headers={"Authorization": "Bearer mock_token"})
    assert response.status_code == 403

def test_token_identity_spoofing_blocked():
    client = TestClient(app)
    
    app.dependency_overrides[verify_firebase_token] = lambda: "legitimate_user"
    
    response = client.get("/token?room=testroom&identity=victim_user", headers={"Authorization": "Bearer mock_token"})
    assert response.status_code == 403

def test_input_validation_large_payload():
    client = TestClient(app)
    app.dependency_overrides[verify_firebase_token] = lambda: "legitimate_user"
    
    payload = {
        "message": "A" * 1000,
        "uid": "legitimate_user"
    }
    response = client.post("/chat", json=payload, headers={"Authorization": "Bearer mock_token"})
    assert response.status_code == 422

def test_input_validation_missing_fields():
    client = TestClient(app)
    app.dependency_overrides[verify_firebase_token] = lambda: "legitimate_user"
    
    payload = {}
    response = client.post("/chat", json=payload, headers={"Authorization": "Bearer mock_token"})
    assert response.status_code == 422

def test_rate_limiting_token_endpoint():
    client = TestClient(app)
    app.dependency_overrides[verify_firebase_token] = lambda: "legitimate_user"
    
    responses = []
    for _ in range(10):
        resp = client.get("/token?room=testroom&identity=legitimate_user", headers={"Authorization": "Bearer mock_token"})
        responses.append(resp.status_code)
        
    assert 429 in responses, "Rate Limiting: Did not receive a 429 Too Many Requests response"
