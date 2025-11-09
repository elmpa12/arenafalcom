import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import importlib


def get_app_module():
    # Ensure repository root is on sys.path
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Disable dotenv before import to avoid loading real .env in tests
    os.environ["DISABLE_DOTENV"] = "1"

    module = importlib.import_module("backend.openai_gateway")
    module.get_settings.cache_clear()
    module._client = None
    return module


def test_health_ok():
    mod = get_app_module()
    client = TestClient(mod.app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_codex_missing_api_key_returns_503(monkeypatch):
    # Ensure no API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mod = get_app_module()
    client = TestClient(mod.app)

    payload = {"prompt": "print('hello')"}
    resp = client.post("/api/codex", json=payload)
    assert resp.status_code == 503
    assert "OPENAI_API_KEY" in resp.json()["detail"]


def test_codex_unauthorized_when_token_required(monkeypatch):
    monkeypatch.setenv("GATEWAY_TOKEN", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mod = get_app_module()
    client = TestClient(mod.app)

    payload = {"prompt": "print('hello')"}
    resp = client.post("/api/codex", json=payload)
    assert resp.status_code == 401

    # Wrong token
    resp = client.post("/api/codex", json=payload, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_codex_unsupported_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GATEWAY_TOKEN", raising=False)

    mod = get_app_module()
    client = TestClient(mod.app)

    payload = {"prompt": "ok", "model": "unknown-model"}
    resp = client.post("/api/codex", json=payload)
    assert resp.status_code == 400
    assert "Unsupported model" in resp.json()["detail"]


def test_codex_happy_path(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GATEWAY_TOKEN", raising=False)

    mod = get_app_module()

    class FakeResponses:
        def create(self, model: str, input: str):  # noqa: A002 - match SDK signature
            class R:
                output_text = "print('ok')\n"
            return R()

    class FakeClient:
        responses = FakeResponses()

    # Patch client factory
    monkeypatch.setattr(mod, "get_openai_client", lambda: FakeClient())

    client = TestClient(mod.app)

    payload = {"prompt": "gerar um hello world"}
    resp = client.post("/api/codex", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"].startswith("print(")


def test_codex_mock_model_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_TOKEN", raising=False)
    mod = get_app_module()
    client = TestClient(mod.app)

    payload = {"prompt": "qualquer", "model": "mock"}
    resp = client.post("/api/codex", json=payload)
    assert resp.status_code == 200
    assert resp.json()["result"].strip() == "print('hello world')"


def test_models_endpoint_without_api_key():
    mod = get_app_module()
    client = TestClient(mod.app)
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert "mock" in data["models"]


def test_models_endpoint_with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mod = get_app_module()

    class M:
        def __init__(self, id):
            self.id = id

    class Page:
        data = [M("gpt-4.1"), M("gpt-4o-mini"), M("text-embedding-3-small")]

    class FakeClient:
        class models:
            @staticmethod
            def list():
                return Page()

    monkeypatch.setattr(mod, "get_openai_client", lambda: FakeClient())
    client = TestClient(mod.app)
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert "mock" in models
    assert "gpt-4.1" in models
    assert "gpt-4o-mini" in models
