
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.approvals import router as approvals_router
from routes.memory import router as memory_router


class AssistantApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_db = os.environ.get("UNIFIED_AI_DB_PATH")
        self.previous_token = os.environ.get("ASSISTANT_API_TOKEN")
        os.environ["UNIFIED_AI_DB_PATH"] = str(
            Path(self.temp_dir.name) / "api.sqlite3"
        )
        self.token = "T" * 48
        os.environ["ASSISTANT_API_TOKEN"] = self.token
        app = FastAPI()
        app.include_router(memory_router)
        app.include_router(approvals_router)
        self.client = TestClient(app)

    def tearDown(self):
        if self.previous_db is None:
            os.environ.pop("UNIFIED_AI_DB_PATH", None)
        else:
            os.environ["UNIFIED_AI_DB_PATH"] = self.previous_db
        if self.previous_token is None:
            os.environ.pop("ASSISTANT_API_TOKEN", None)
        else:
            os.environ["ASSISTANT_API_TOKEN"] = self.previous_token
        self.temp_dir.cleanup()

    def test_memory_api_requires_token(self):
        response = self.client.get("/assistant/memory")
        self.assertEqual(response.status_code, 401)

    def test_memory_api_create_list_and_delete(self):
        headers = {"X-Assistant-Token": self.token}
        created = self.client.post(
            "/assistant/memory",
            headers=headers,
            json={
                "category": "profile",
                "memory_key": "education",
                "memory_value": "B.Tech AI/ML; graduating 2029",
            },
        )
        self.assertEqual(created.status_code, 201)
        memory_id = created.json()["memory_id"]
        listed = self.client.get("/assistant/memory", headers=headers)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["count"], 1)
        deleted = self.client.delete(
            f"/assistant/memory/{memory_id}", headers=headers
        )
        self.assertEqual(deleted.status_code, 204)

    def test_approval_api_never_returns_action_payload(self):
        headers = {"X-Assistant-Token": self.token}
        created = self.client.post(
            "/assistant/approvals",
            headers=headers,
            json={
                "action_type": "gmail.send",
                "summary": "Send the reviewed draft to hiring@example.com.",
                "action_payload": {
                    "to": "hiring@example.com",
                    "subject": "Application",
                    "body": "private draft content",
                },
            },
        )
        self.assertEqual(created.status_code, 201)
        self.assertNotIn("action_payload", created.json())
        approval_id = created.json()["approval_id"]
        decision = self.client.post(
            f"/assistant/approvals/{approval_id}/decision",
            headers=headers,
            json={"decision": "approved", "reason": "Reviewed by user"},
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["status"], "approved")
        self.assertNotIn("action_payload", decision.json())


if __name__ == "__main__":
    unittest.main()


