"""Unit tests for TaskStore."""

import shutil
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from datus.storage.task.store import TaskStore


class TestTaskStore:
    """Test cases for TaskStore."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def store(self, temp_dir):
        return TaskStore(temp_dir)

    def test_create_and_get_task(self, store):
        created = store.create_task("task-1", "select 1")
        assert created["task_id"] == "task-1"
        assert created["task_query"] == "select 1"

        fetched = store.get_task("task-1")
        assert fetched is not None
        assert fetched["task_id"] == "task-1"
        assert fetched["task_query"] == "select 1"

    def test_update_task(self, store):
        store.create_task("task-2", "select 2")
        updated = store.update_task("task-2", sql_query="SELECT 2", sql_result="2", status="success")
        assert updated is True

        fetched = store.get_task("task-2")
        assert fetched is not None
        assert fetched["sql_query"] == "SELECT 2"
        assert fetched["sql_result"] == "2"
        assert fetched["status"] == "success"

    def test_record_and_get_feedback(self, store):
        store.create_task("task-3", "select 3")
        recorded = store.record_feedback("task-3", "success")
        assert recorded["task_id"] == "task-3"
        assert recorded["user_feedback"] == "success"

        feedback = store.get_feedback("task-3")
        assert feedback is not None
        assert feedback["user_feedback"] == "success"

    def test_get_all_feedback(self, store):
        store.create_task("task-4", "select 4")
        store.create_task("task-5", "select 5")
        store.record_feedback("task-4", "success")
        store.record_feedback("task-5", "failed")

        feedback = store.get_all_feedback()
        assert len(feedback) == 2
        assert {item["task_id"] for item in feedback} == {"task-4", "task-5"}

    def test_delete_feedback(self, store):
        store.create_task("task-6", "select 6")
        store.record_feedback("task-6", "success")
        cleared = store.delete_feedback("task-6")
        assert cleared is True

        feedback = store.get_feedback("task-6")
        assert feedback is None

    def test_cleanup_old_tasks(self, store):
        store.create_task("task-7", "select 7")
        store.create_task("task-8", "select 8")

        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat().replace("+00:00", "Z")
        store.table.update(where=None, values={"created_at": old_time})

        deleted = store.cleanup_old_tasks(hours=24)
        assert deleted >= 2
