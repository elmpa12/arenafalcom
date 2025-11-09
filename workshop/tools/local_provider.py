from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .providers import InstanceMeta


class LocalProvider:
    """No-op provider that represents an already available host."""

    provider_name = "local"

    def launch(
        self,
        *,
        host: str,
        name: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **_: Any,
    ) -> InstanceMeta:
        if not host:
            raise SystemExit("Provider 'local' requer --host com IP ou hostname acessível.")

        region_value = (region or "local").strip() or "local"
        timestamp = datetime.now(timezone.utc).isoformat()
        instance_id = f"local:{host}"

        return InstanceMeta(
            provider="local",
            region=region_value,
            instance_id=instance_id,
            state="available",
            public_ip=host,
            private_ip=host,
            name=name or "Local GPU",
            launch_time=timestamp,
            tags=dict(tags or {}),
        )

    def reuse(
        self,
        *,
        host: Optional[str] = None,
        name: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Optional[InstanceMeta]:
        if not host:
            return None
        return self.launch(host=host, name=name, region=region, tags=tags, **kwargs)

    def terminate(self, meta: InstanceMeta) -> None:  # pragma: no cover - apenas log
        print(
            "Provider 'local': nada para terminar (instância persistente)."
            f" Identificador: {meta.instance_id}"
        )
