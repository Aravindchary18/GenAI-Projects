
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.memory_service import (
    delete_memory,
    get_memory,
    list_memories,
    save_memory,
    update_memory,
)


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "assistant.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_memory(self):
        saved = save_memory(
            category="profile",
            memory_key="education",
            memory_value="B.Tech AI/ML, graduating 2029",
            owner_id="test-user",
            db_path=self.db_path,
        )
        loaded = get_memory(
            saved["memory_id"], owner_id="test-user", db_path=self.db_path
        )
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["memory_value"], "B.Tech AI/ML, graduating 2029")

    def test_same_key_upserts_instead_of_duplicating(self):
        first = save_memory(
            category="profile", memory_key="Name", memory_value="Alex",
            owner_id="test-user", db_path=self.db_path,
        )
        second = save_memory(
            category="profile", memory_key=" name ", memory_value="Alex Updated",
            owner_id="test-user", db_path=self.db_path,
        )
        self.assertEqual(first["memory_id"], second["memory_id"])
        self.assertEqual(second["memory_value"], "Alex Updated")
        self.assertEqual(
            len(list_memories(owner_id="test-user", db_path=self.db_path)), 1
        )

    def test_search_and_category_filter(self):
        save_memory(
            category="profile", memory_key="degree", memory_value="Artificial Intelligence",
            owner_id="test-user", db_path=self.db_path,
        )
        save_memory(
            category="preference", memory_key="work mode", memory_value="Remote",
            owner_id="test-user", db_path=self.db_path,
        )
        results = list_memories(
            query="Artificial", category="profile",
            owner_id="test-user", db_path=self.db_path,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["memory_key"], "degree")

    def test_update_changes_existing_record(self):
        saved = save_memory(
            category="profile", memory_key="city", memory_value="Surat",
            owner_id="test-user", db_path=self.db_path,
        )
        updated = update_memory(
            saved["memory_id"], category="profile", memory_key="city",
            memory_value="Vadodara", owner_id="test-user", db_path=self.db_path,
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["memory_value"], "Vadodara")

    def test_owner_cannot_read_or_delete_another_owners_memory(self):
        saved = save_memory(
            category="profile", memory_key="email", memory_value="private@example.com",
            owner_id="owner-a", db_path=self.db_path,
        )
        self.assertIsNone(
            get_memory(saved["memory_id"], owner_id="owner-b", db_path=self.db_path)
        )
        self.assertFalse(
            delete_memory(saved["memory_id"], owner_id="owner-b", db_path=self.db_path)
        )

    def test_forget_deletes_memory(self):
        saved = save_memory(
            category="preference", memory_key="theme", memory_value="dark",
            owner_id="test-user", db_path=self.db_path,
        )
        self.assertTrue(
            delete_memory(saved["memory_id"], owner_id="test-user", db_path=self.db_path)
        )
        self.assertIsNone(
            get_memory(saved["memory_id"], owner_id="test-user", db_path=self.db_path)
        )

    def test_rejects_empty_values_and_unsupported_categories(self):
        with self.assertRaises(ValueError):
            save_memory(
                category="profile", memory_key=" ", memory_value="value",
                owner_id="test-user", db_path=self.db_path,
            )
        with self.assertRaises(ValueError):
            save_memory(
                category="secret-category", memory_key="key", memory_value="value",
                owner_id="test-user", db_path=self.db_path,
            )


if __name__ == "__main__":
    unittest.main()

