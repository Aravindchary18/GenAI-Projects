
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.assistant import router


class AssistantWorkflowApiTests(unittest.TestCase):
    def setUp(self):
        self.previous_token = os.environ.get("ASSISTANT_API_TOKEN")
        os.environ["ASSISTANT_API_TOKEN"] = "test-token-for-assistant-api-32-chars"
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.headers = {"X-Assistant-Token": os.environ["ASSISTANT_API_TOKEN"]}

    def tearDown(self):
        if self.previous_token is None:
            os.environ.pop("ASSISTANT_API_TOKEN", None)
        else:
            os.environ["ASSISTANT_API_TOKEN"] = self.previous_token

    def test_token_is_required(self):
        response = self.client.post("/assistant/tasks", json={"user_request": "Research current facts"})
        self.assertEqual(response.status_code, 401)

    def test_start_task_passes_validated_request_to_graph(self):
        state = {"thread_id": "thread-123", "task_status": "completed", "final_response": "Done", "requested_specialists": ["research"]}
        with patch("routes.assistant._load_graph_functions", return_value=(lambda *a, **k: state, lambda *_: None, lambda *_a, **_k: state)) as loader:
            response = self.client.post("/assistant/tasks", headers=self.headers, json={"user_request": "Research current facts"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["thread_id"], "thread-123")
        self.assertEqual(response.json()["final_response"], "Done")
        loader.assert_called_once()

    def test_read_missing_task_returns_404(self):
        with patch("routes.assistant._load_graph_functions", return_value=(lambda *a, **k: {}, lambda *_: None, lambda *_a, **_k: {})):
            response = self.client.get("/assistant/tasks/not-found", headers=self.headers)
        self.assertEqual(response.status_code, 404)

    def test_bad_thread_id_is_rejected(self):
        response = self.client.get("/assistant/tasks/../../bad", headers=self.headers)
        self.assertIn(response.status_code, {404, 422})


if __name__ == "__main__":
    unittest.main()
