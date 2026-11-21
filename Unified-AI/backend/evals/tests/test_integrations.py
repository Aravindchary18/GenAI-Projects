
from __future__ import annotations

import base64
import email
import email.policy
import unittest
from unittest.mock import patch

from services.gmail_service import build_raw_message, send_draft_after_approval
from tools.browser_tools import _safe_filename, validate_public_url


class BrowserSafetyTests(unittest.TestCase):
    def test_rejects_non_http_schemes(self):
        with self.assertRaises(ValueError):
            validate_public_url("file:///etc/passwd", resolve_dns=False)

    def test_rejects_localhost(self):
        with self.assertRaises(ValueError):
            validate_public_url("http://localhost:8000/", resolve_dns=False)

    def test_rejects_private_ip(self):
        with self.assertRaises(ValueError):
            validate_public_url("http://192.168.1.10/", resolve_dns=False)

    def test_rejects_url_embedded_credentials(self):
        with self.assertRaises(ValueError):
            validate_public_url("https://user:password@example.com/", resolve_dns=False)

    def test_accepts_global_ip_address(self):
        self.assertEqual(
            validate_public_url("https://8.8.8.8/", resolve_dns=False),
            "https://8.8.8.8/",
        )

    def test_filename_is_sanitized_and_pdf_extension_added(self):
        self.assertEqual(_safe_filename("../../study notes"), "study_notes.pdf")


class GmailMessageTests(unittest.TestCase):
    def test_builds_url_safe_raw_message(self):
        raw = build_raw_message("person@example.com", "Project update", "The draft is ready.")
        decoded = base64.urlsafe_b64decode(raw.encode("ascii"))
        message = email.message_from_bytes(decoded, policy=email.policy.default)
        self.assertEqual(message["To"], "person@example.com")
        self.assertEqual(message["Subject"], "Project update")
        self.assertIn("The draft is ready.", message.get_content())

    def test_rejects_header_injection(self):
        with self.assertRaises(ValueError):
            build_raw_message("person@example.com\nBcc: attacker@example.com", "hello", "body")

    def test_rejects_invalid_recipient(self):
        with self.assertRaises(ValueError):
            build_raw_message("not-an-email", "hello", "body")


class GmailApprovalGateTests(unittest.TestCase):
    @patch("services.approval_service.get_approval", return_value=None)
    def test_send_requires_existing_approval(self, get_approval):
        with self.assertRaises(PermissionError):
            send_draft_after_approval(draft_id="draft-123", approval_id="approval-123")
        get_approval.assert_called_once_with("approval-123", include_payload=True)

    @patch(
        "services.approval_service.get_approval",
        return_value={
            "status": "approved",
            "action_type": "gmail.send",
            "action_payload": {"draft_id": "another-draft"},
        },
    )
    def test_send_rejects_approval_for_different_draft(self, _get_approval):
        with self.assertRaises(PermissionError):
            send_draft_after_approval(draft_id="draft-123", approval_id="approval-123")

    @patch(
        "services.approval_service.get_approval",
        return_value={
            "status": "pending",
            "action_type": "gmail.send",
            "action_payload": {"draft_id": "draft-123"},
        },
    )
    def test_send_rejects_pending_approval(self, _get_approval):
        with self.assertRaises(PermissionError):
            send_draft_after_approval(draft_id="draft-123", approval_id="approval-123")


if __name__ == "__main__":
    unittest.main()
