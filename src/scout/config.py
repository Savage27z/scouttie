"""Configuration loading: environment, targets, filter thresholds.

Everything tunable lives in config/*.yaml rather than in code, so that
`scout replay` can re-score stored history against new thresholds without
touching the API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = PROJECT_ROOT / "config" / "targets.yaml"
DEFAULT_FILTERS = PROJECT_ROOT / "config" / "filters.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


@dataclass(frozen=True)
class TargetSpec:
    """One ecosystem account to watch for new followers."""

    username: str
    active: bool = True
    poll_interval_s: int = 3600
    page_size: int = 200
    max_pages_per_poll: int = 10
    # Bounds for the adaptive poll interval. The cap is detection latency: how long a
    # new project can sit unnoticed. Polling costs ~$0.002, so latency is cheap to buy.
    min_poll_interval_s: int = 900
    max_poll_interval_s: int = 7200

    def __post_init__(self) -> None:
        if not self.username or self.username.startswith("@"):
            raise ConfigError(
                f"target username must be a bare handle without '@': {self.username!r}"
            )
        if not 20 <= self.page_size <= 200:
            raise ConfigError(
                f"{self.username}: page_size must be 20-200 (provider limit), "
                f"got {self.page_size}"
            )
        if self.max_pages_per_poll < 1:
            raise ConfigError(f"{self.username}: max_pages_per_poll must be >= 1")
        if not 60 <= self.min_poll_interval_s <= self.max_poll_interval_s:
            raise ConfigError(
                f"{self.username}: need 60 <= min_poll_interval_s <= max_poll_interval_s, "
                f"got {self.min_poll_interval_s}..{self.max_poll_interval_s}"
            )


@dataclass(frozen=True)
class Secrets:
    """Credentials from the environment. Absent values are empty strings, so that
    commands which do not need a given credential still run."""

    twitterapi_key: str = ""
    deepseek_api_key: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""

    def require(self, *names: str) -> None:
        """Fail loudly and early when a command needs a credential it lacks."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            env_names = {
                "twitterapi_key": "TWITTERAPI_KEY",
                "deepseek_api_key": "DEEPSEEK_API_KEY",
                "telegram_token": "TELEGRAM_TOKEN",
                "telegram_chat_id": "TELEGRAM_CHAT_ID",
            }
            pretty = ", ".join(env_names.get(n, n.upper()) for n in missing)
            raise ConfigError(
                f"missing required credential(s): {pretty}. "
                f"Copy .env.example to .env and fill them in."
            )


@dataclass(frozen=True)
class Config:
    secrets: Secrets
    targets: tuple[TargetSpec, ...]
    filters: dict[str, Any] = field(default_factory=dict)
    db_path: Path = PROJECT_ROOT / "data" / "scout.db"
    leads_dir: Path = PROJECT_ROOT / "leads"
    log_level: str = "INFO"

    def active_targets(self) -> tuple[TargetSpec, ...]:
        return tuple(t for t in self.targets if t.active)

    def stage1(self) -> dict[str, Any]:
        return self.filters.get("stage1", {})

    def stage2(self) -> dict[str, Any]:
        return self.filters.get("stage2", {})

    def stage3(self) -> dict[str, Any]:
        return self.filters.get("stage3", {})

    def llm(self) -> dict[str, Any]:
        return self.filters.get("llm", {})

    def alerts(self) -> dict[str, Any]:
        return self.filters.get("alerts", {})


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_targets(path: Path = DEFAULT_TARGETS) -> tuple[TargetSpec, ...]:
    raw = _load_yaml(path)
    defaults = raw.get("defaults") or {}
    entries = raw.get("targets")
    if not entries:
        raise ConfigError(f"{path} defines no targets")

    specs: list[TargetSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            entry = {"username": entry}
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: each target must be a mapping or string, got {entry!r}")
        merged = {**defaults, **entry}
        spec = TargetSpec(
            username=str(merged.get("username", "")).lstrip("@"),
            active=bool(merged.get("active", True)),
            poll_interval_s=int(merged.get("poll_interval_s", 3600)),
            page_size=int(merged.get("page_size", 200)),
            max_pages_per_poll=int(merged.get("max_pages_per_poll", 10)),
            min_poll_interval_s=int(merged.get("min_poll_interval_s", 900)),
            max_poll_interval_s=int(merged.get("max_poll_interval_s", 7200)),
        )
        key = spec.username.lower()
        if key in seen:
            raise ConfigError(f"{path}: duplicate target {spec.username!r}")
        seen.add(key)
        specs.append(spec)
    return tuple(specs)


def load_config(
    targets_path: Path = DEFAULT_TARGETS,
    filters_path: Path = DEFAULT_FILTERS,
) -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    secrets = Secrets(
        twitterapi_key=os.getenv("TWITTERAPI_KEY", "").strip(),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        telegram_token=os.getenv("TELEGRAM_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )
    db_path = Path(os.getenv("SCOUT_DB_PATH") or (PROJECT_ROOT / "data" / "scout.db"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Config(
        secrets=secrets,
        targets=load_targets(targets_path),
        filters=_load_yaml(filters_path),
        db_path=db_path,
        leads_dir=PROJECT_ROOT / "leads",
        log_level=os.getenv("SCOUT_LOG_LEVEL", "INFO").upper(),
    )
