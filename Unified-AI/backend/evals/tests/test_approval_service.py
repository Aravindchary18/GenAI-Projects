
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.approval_service import (
    ApprovalAlreadyDecidedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    create_approval,
    decide_approval,
    get_approval,
    list_approvals,
)
from services.assistant_store import database_connection


class ApprovalServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "assistant.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create(self, owner_id="test-user"):
        return create_approval(
            action_type="gmail.send",
            summary="Send the reviewed email to the specified recipient.",
            action_payload={"to": "person@example.com", "subject": "Hello"},
            owner_id=owner_id,
            db_path=self.db_path,
        )

    def test_create_is_pending_and_payload_is_internal(self):
        created = self._create()
        self.assertEqual(created["status"], "pending")
        self.assertEqual(created["action_payload"]["to"], "person@example.com")
        public_record = get_approval(
            created["approval_id"], include_payload=False,
            owner_id="test-user", db_path=self.db_path,
        )
        self.assertNotIn("action_payload", public_record)

    def test_approval_can_be_decided_only_once(self):
        created = self._create()
        decided = decide_approval(
            created["approval_id"], decision="approved",
            owner_id="test-user", db_path=self.db_path,
        )
        self.assertEqual(decided["status"], "approved")
        with self.assertRaises(ApprovalAlreadyDecidedError):
            decide_approval(
                created["approval_id"], decision="rejected",
                owner_id="test-user", db_path=self.db_path,
            )

    def test_rejection_is_terminal(self):
        created = self._create()
        decided = decide_approval(
            created["approval_id"], decision="rejected", reason="Not approved.",
            owner_id="test-user", db_path=self.db_path,
        )
        self.assertEqual(decided["status"], "rejected")

    def test_expired_approval_cannot_be_approved(self):
        created = self._create()
        expired_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(
            timespec="microseconds"
        )
        with database_connection(self.db_path) as connection:
            connection.execute(
                "UPDATE assistant_action_approvals SET expires_at = ? WHERE approval_id = ?",
                (expired_at, created["approval_id"]),
            )
        with self.assertRaises(ApprovalExpiredError):
            decide_approval(
                created["approval_id"], decision="approved",
                owner_id="test-user", db_path=self.db_path,
            )
        record = get_approval(
            created["approval_id"], owner_id="test-user", db_path=self.db_path
        )
        self.assertEqual(record["status"], "expired")

    def test_owner_isolation(self):
        created = self._create(owner_id="owner-a")
        self.assertIsNone(
            get_approval(
                created["approval_id"], owner_id="owner-b", db_path=self.db_path
            )
        )
        with self.assertRaises(ApprovalNotFoundError):
            decide_approval(
                created["approval_id"], decision="approved",
                owner_id="owner-b", db_path=self.db_path,
            )

    def test_list_returns_only_requested_status(self):
        pending = self._create()
        approved = self._create()
        decide_approval(
            approved["approval_id"], decision="approved",
            owner_id="test-user", db_path=self.db_path,
        )
        self.assertEqual(
            [x["approval_id"] for x in list_approvals(
                status="pending", owner_id="test-user", db_path=self.db_path
            )],
            [pending["approval_id"]],
        )

    def test_rejects_invalid_action_and_payload(self):
        with self.assertRaises(ValueError):
            create_approval(
                action_type="Send Email", summary="send", action_payload={},
                owner_id="test-user", db_path=self.db_path,
            )
        with self.assertRaises(ValueError):
            create_approval(
                action_type="gmail.send", summary="send", action_payload={"bad": object()},
                owner_id="test-user", db_path=self.db_path,
            )


if __name__ == "__main__":
    unittest.main()
