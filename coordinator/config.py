"""Coordinator configuration loaded from environment variables (the .env file on box1).

Alpaca keys are read here and only here; nothing else in the repository sees them,
and no worker ever receives them."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent


def _as_bool(value: str) -> bool:
    """Interpret common truthy strings."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_origins(value: str) -> tuple[str, ...]:
    """Split a comma separated origin list, normalising each entry."""
    return tuple(normalise_origin(part) for part in value.split(",") if part.strip())


def normalise_origin(origin: str) -> str:
    """Lowercase an origin and strip whitespace and trailing slashes."""
    return origin.strip().lower().rstrip("/")


@dataclass(frozen=True)
class Config:
    """Runtime settings for the host process."""

    database_url: str = "postgresql://fleet2:fleet2@db:5432/fleet2"
    public_url: str = "http://127.0.0.1:8090"
    owner_login: str = ""
    dev: bool = False
    allowed_origins: tuple[str, ...] = field(default_factory=tuple)
    bind: str = "127.0.0.1:8090"
    deploy_dir: Path = REPO_ROOT / "deploy"
    loop_seconds: float = 5.0
    allow_worker_ips: bool = False
    # The host sits behind tailscale serve, which appends the tailnet peer IP as the
    # last X-Forwarded-For hop. Set FLEET_TRUST_PROXY=0 when clients hit the port directly.
    trust_proxy: bool = True
    limits_path: Path = REPO_ROOT / "config" / "limits.toml"
    # Alpaca: paper keys by default. Live needs ALPACA_LIVE=true here AND the owner's
    # typed confirmation in the dashboard; neither alone is enough.
    alpaca_paper_key_id: str = field(default="", repr=False)
    alpaca_paper_secret: str = field(default="", repr=False)
    alpaca_live_allowed: bool = False
    alpaca_live_key_id: str = field(default="", repr=False)
    alpaca_live_secret: str = field(default="", repr=False)
    fake_broker: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_origins:
            object.__setattr__(self, "allowed_origins", (normalise_origin(self.public_url),))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        """Build a Config from FLEET_* environment variables."""
        env = os.environ if env is None else env
        public_url = env.get("FLEET_PUBLIC_URL", cls.public_url).rstrip("/")
        origins = env.get("FLEET_ALLOWED_ORIGINS", "").strip()
        return cls(
            database_url=env.get("DATABASE_URL", cls.database_url),
            public_url=public_url,
            owner_login=env.get("FLEET_OWNER_LOGIN", "").strip(),
            dev=_as_bool(env.get("FLEET_DEV", "")),
            allowed_origins=_split_origins(origins) if origins else (normalise_origin(public_url),),
            bind=env.get("FLEET_BIND", cls.bind),
            deploy_dir=Path(env.get("FLEET_DEPLOY_DIR", str(REPO_ROOT / "deploy"))),
            loop_seconds=float(env.get("FLEET_LOOP_SECONDS", "5")),
            allow_worker_ips=_as_bool(env.get("FLEET_OWNER_ALLOW_WORKER_IPS", "")),
            trust_proxy=_as_bool(env.get("FLEET_TRUST_PROXY", "1")),
            limits_path=Path(env.get("FLEET_LIMITS_FILE", str(REPO_ROOT / "config" / "limits.toml"))),
            alpaca_paper_key_id=env.get("ALPACA_PAPER_KEY_ID", "").strip(),
            alpaca_paper_secret=env.get("ALPACA_PAPER_SECRET_KEY", "").strip(),
            alpaca_live_allowed=_as_bool(env.get("ALPACA_LIVE", "")),
            alpaca_live_key_id=env.get("ALPACA_LIVE_KEY_ID", "").strip(),
            alpaca_live_secret=env.get("ALPACA_LIVE_SECRET_KEY", "").strip(),
            fake_broker=_as_bool(env.get("FLEET_FAKE_BROKER", "")),
        )

    @property
    def bind_host(self) -> str:
        """Host part of FLEET_BIND."""
        host, _, _ = self.bind.rpartition(":")
        return host or "127.0.0.1"

    @property
    def bind_port(self) -> int:
        """Port part of FLEET_BIND."""
        _, _, port = self.bind.rpartition(":")
        return int(port or "8090")
