from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class FirestoreConversationStore:
    def __init__(self, client):
        self.client = client

    def save_message(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        question: str,
        answer: str,
        audio: dict[str, Any] | None,
    ) -> None:
        conversation_ref = self.client.collection("conversations").document(conversation_id)
        conversation_ref.set(
            {
                "userId": user_id,
                "conversationId": conversation_id,
                "updatedAt": datetime.now(timezone.utc),
            },
            merge=True,
        )
        conversation_ref.collection("messages").document(message_id).set(
            {
                "userId": user_id,
                "conversationId": conversation_id,
                "question": question,
                "answer": answer,
                "audio": audio,
                "createdAt": datetime.now(timezone.utc),
            }
        )

    def save_analytics_transcript(
        self,
        user_id: str,
        question: str,
        answer: str,
        is_answered: bool,
    ) -> None:
        self.client.collection("transcripts").add(
            {
                "uid": user_id,
                "question": question,
                "answer": answer,
                "is_answered": is_answered,
                "createdAt": datetime.now(timezone.utc),
                "metadata": {"channel": "chat"},
            }
        )
