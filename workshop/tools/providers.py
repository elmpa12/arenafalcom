from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

DEFAULT_METADATA_PATH = Path("tools/last_gpu.json")


@dataclass
class SSHConfig:
    host: str
    user: str
    key_path: str
    port: int = 22


@dataclass
class InstanceMeta:
    provider: str
    region: str
    instance_id: str
    state: str
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    name: Optional[str] = None
    launch_time: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tags"] = dict(self.tags or {})
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InstanceMeta":
        tags = payload.get("tags") or {}
        return cls(
            provider=payload["provider"],
            region=payload["region"],
            instance_id=payload["instance_id"],
            state=payload.get("state", "unknown"),
            public_ip=payload.get("public_ip"),
            private_ip=payload.get("private_ip"),
            name=payload.get("name"),
            launch_time=payload.get("launch_time"),
            tags=dict(tags),
        )


class Provider(Protocol):
    def launch(self, **kwargs: Any) -> InstanceMeta:
        ...

    def reuse(self, **kwargs: Any) -> Optional[InstanceMeta]:
        ...

    def terminate(self, meta: InstanceMeta) -> None:
        ...


def provider_factory(name: str) -> Provider:
    normalized = name.lower().strip()
    if normalized == "aws":
        from .aws_provider import AWSProvider

        return AWSProvider()
    if normalized == "local":
        from .local_provider import LocalProvider

        return LocalProvider()
    raise ValueError(f"unknown provider '{name}'")


def save_metadata(meta: InstanceMeta, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_metadata(path: Path) -> InstanceMeta:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return InstanceMeta.from_dict(payload)
