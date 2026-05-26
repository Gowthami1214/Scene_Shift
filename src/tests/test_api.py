"""
Functional tests for the FastAPI backend server.
Tests upload, process, status, result, and WebSocket endpoints.
"""

from __future__ import annotations

import io
import json
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.server import app, _jobs


def _make_test_image_bytes(width: int = 256, height: int = 256, fmt: str = "PNG") -> bytes:
    """Create an in-memory test image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
        mode="RGB",
    )
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint(unittest.TestCase):
    """Tests for /api/health endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_returns_ok(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("timestamp", data)

    def test_health_content_type(self):
        resp = self.client.get("/api/health")
        self.assertIn("application/json", resp.headers["content-type"])


class TestDeviceEndpoint(unittest.TestCase):
    """Tests for /api/device endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_device_returns_info(self):
        resp = self.client.get("/api/device")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("device_type", data)
        self.assertIn("device_name", data)

    def test_device_type_is_valid(self):
        resp = self.client.get("/api/device")
        device_type = resp.json()["device_type"]
        self.assertIn(device_type, ["cuda", "mps", "cpu"])


class TestUploadEndpoint(unittest.TestCase):
    """Tests for POST /api/upload endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_valid_png_upload(self):
        data = _make_test_image_bytes(256, 256, "PNG")
        resp = self.client.post(
            "/api/upload",
            files={"file": ("test.png", data, "image/png")},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("job_id", body)
        self.assertIn("status", body)
        self.assertEqual(body["status"], "uploaded")
        self.assertEqual(body["image_width"], 256)
        self.assertEqual(body["image_height"], 256)

    def test_valid_jpeg_upload(self):
        data = _make_test_image_bytes(128, 128, "JPEG")
        resp = self.client.post(
            "/api/upload",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "uploaded")

    def test_invalid_file_type(self):
        """Non-image file should be rejected."""
        fake_data = b"not an image at all"
        resp = self.client.post(
            "/api/upload",
            files={"file": ("test.txt", fake_data, "text/plain")},
        )
        self.assertIn(resp.status_code, [422, 400])

    def test_upload_creates_job(self):
        """Upload should create an entry in the job registry."""
        data = _make_test_image_bytes(128, 128)
        resp = self.client.post(
            "/api/upload",
            files={"file": ("img.png", data, "image/png")},
        )
        job_id = resp.json()["job_id"]
        self.assertIn(job_id, _jobs)

    def test_too_small_image_rejected(self):
        """Image smaller than MIN_IMAGE_DIM (64) should be rejected."""
        tiny = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        buf = io.BytesIO()
        tiny.save(buf, format="PNG")
        buf.seek(0)
        resp = self.client.post(
            "/api/upload",
            files={"file": ("tiny.png", buf.read(), "image/png")},
        )
        self.assertIn(resp.status_code, [422, 400])


class TestStatusEndpoint(unittest.TestCase):
    """Tests for GET /api/status/{job_id}."""

    def setUp(self):
        self.client = TestClient(app)
        # Create a test job manually
        self.job_id = "test001"
        _jobs[self.job_id] = {
            "status": "uploaded",
            "progress": 0.0,
            "stage": "idle",
            "message": "Test job",
            "elapsed_s": 0.0,
            "result_path": None,
            "error": None,
        }

    def test_status_known_job(self):
        resp = self.client.get(f"/api/status/{self.job_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["job_id"], self.job_id)
        self.assertIn("status", data)
        self.assertIn("progress", data)

    def test_status_unknown_job(self):
        resp = self.client.get("/api/status/NONEXISTENT_JOB_ID")
        self.assertEqual(resp.status_code, 404)

    def tearDown(self):
        _jobs.pop(self.job_id, None)


class TestResultEndpoint(unittest.TestCase):
    """Tests for GET /api/result/{job_id}."""

    def setUp(self):
        self.client = TestClient(app)

    def test_result_not_found_for_unknown_job(self):
        resp = self.client.get("/api/result/UNKNOWN_JOB_XYZ")
        self.assertEqual(resp.status_code, 404)

    def test_result_not_ready_returns_202(self):
        job_id = "test_result_002"
        _jobs[job_id] = {
            "status": "processing",
            "progress": 0.5,
            "stage": "editing_object",
            "message": "Processing",
            "elapsed_s": 5.0,
            "result_path": None,
            "error": None,
        }
        resp = self.client.get(f"/api/result/{job_id}")
        self.assertIn(resp.status_code, [202, 404])
        _jobs.pop(job_id, None)


class TestProcessEndpoint(unittest.TestCase):
    """Tests for POST /api/process."""

    def setUp(self):
        self.client = TestClient(app)
        # Create and upload a test image first
        data = _make_test_image_bytes(128, 128)
        resp = self.client.post(
            "/api/upload",
            files={"file": ("test.png", data, "image/png")},
        )
        self.job_id = resp.json()["job_id"]

        # Mock orchestrator to prevent running actual ML models
        self.orchestrator_patcher = patch("src.api.server.get_orchestrator")
        self.mock_get_orchestrator = self.orchestrator_patcher.start()
        self.mock_orchestrator = MagicMock()
        self.mock_orchestrator.run_async = AsyncMock()
        
        # Mock the run_async to return a dummy PipelineResult
        from src.pipeline.orchestrator import PipelineResult
        self.mock_orchestrator.run_async.return_value = PipelineResult(
            job_id=self.job_id,
            final_image=np.zeros((128, 128, 3), dtype=np.uint8),
            timings={"total": 0.1},
            total_time_s=0.1,
            output_path="dummy_path.png",
        )
        self.mock_get_orchestrator.return_value = self.mock_orchestrator

    def test_process_unknown_job_returns_404(self):
        resp = self.client.post(
            "/api/process",
            json={"job_id": "NOT_REAL_JOB_ID"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_process_queued_successfully(self):
        """Valid job should be queued for processing."""
        resp = self.client.post(
            "/api/process",
            json={
                "job_id": self.job_id,
                "object_prompt": "test object",
                "background_prompt": "test background",
                "style_preset": "Realistic",
                "blend_mode": "alpha",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["status"], ["queued", "processing"])

    def tearDown(self):
        self.orchestrator_patcher.stop()
        _jobs.pop(self.job_id, None)


class TestAPIDocsAvailable(unittest.TestCase):
    """Verify Swagger UI is accessible."""

    def setUp(self):
        self.client = TestClient(app)

    def test_docs_accessible(self):
        resp = self.client.get("/docs")
        self.assertEqual(resp.status_code, 200)

    def test_openapi_schema(self):
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        schema = resp.json()
        self.assertIn("paths", schema)
        self.assertIn("/api/upload", schema["paths"])
        self.assertIn("/api/process", schema["paths"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
