"""Unit tests for FeedbackStore."""

import shutil
import tempfile

import pytest

from datus.storage.feedback.store import FeedbackStore


class TestFeedbackStore:
    """Test cases for FeedbackStore."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def store(self, temp_dir):
        return FeedbackStore(temp_dir)

    def test_record_and_get_feedback(self, store):
        recorded = store.record_feedback("task-1", "success")
        assert recorded["task_id"] == "task-1"
        assert recorded["status"] == "success"

        fetched = store.get_feedback("task-1")
        assert fetched is not None
        assert fetched["task_id"] == "task-1"
        assert fetched["status"] == "success"

    def test_get_all_feedback(self, store):
        store.record_feedback("task-2", "success")
        store.record_feedback("task-3", "failed")
        feedback = store.get_all_feedback()
        assert len(feedback) == 2
        assert {row["task_id"] for row in feedback} == {"task-2", "task-3"}

    def test_delete_feedback(self, store):
        store.record_feedback("task-4", "success")
        deleted = store.delete_feedback("task-4")
        assert deleted is True
        assert store.get_feedback("task-4") is None
