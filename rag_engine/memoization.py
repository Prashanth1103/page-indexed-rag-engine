import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


class StageMemoCache:
    """Deterministic disk-backed JSON stage cache preventing redundant LLM calls across agent steps."""

    def __init__(self, cache_dir: str = ".stage_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def compute_key(stage: str, payload: dict[str, Any]) -> str:
        raw = json.dumps({"stage": stage, "payload": payload}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, stage: str, payload: dict[str, Any]) -> Optional[Any]:
        key = self.compute_key(stage, payload)
        path = self.cache_dir / f"{stage}_{key[:16]}.json"
        if path.exists():
            self.hits += 1
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)["data"]
        self.misses += 1
        return None

    def set(self, stage: str, payload: dict[str, Any], data: Any) -> str:
        key = self.compute_key(stage, payload)
        path = self.cache_dir / f"{stage}_{key[:16]}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stage": stage, "key": key, "data": data}, f, indent=2)
        return key
