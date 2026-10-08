"""Pending posts: the /complete or /fail call of a finished job, plus the posts that
must precede it (docs/PROTOCOL.md, step 3).

A result carrying create_models becomes a sequence: one POST /api/v1/models per entry
(in order), then for a backtest job with params.model_id one POST
/api/v1/models/{id}/backtest (for a validate job one POST /api/v1/models/{id}/validation
with its validation_metrics and stress_metrics), then the /complete whose result
carries created_models instead of create_models. The sequence is persisted in pending_posts.json with the
index of the next step, so a crash between posts resumes where it stopped and never
re-posts a model the host already acknowledged (the host is idempotent on the model
identity as well). A step the host refuses (4xx) turns the sequence into a /fail with
a clear error; a 5xx or no answer is retried like any pending post.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from fleet2.common import http

log = logging.getLogger("fleet2.agent")

MODEL_FIELDS = ("family", "params", "artifact", "backtest_metrics", "summary", "parent_model_id", "trained_through",
                "validation_metrics", "stress_metrics")
VALIDATION_FIELDS = ("validation_metrics", "stress_metrics")
MODELS_PATH = "/api/v1/models"


@dataclass
class PendingPost:
    """A /complete or /fail call that has not been acknowledged yet, with the model
    posts (steps) the host must accept first."""

    path: str
    body: dict[str, Any]
    job_id: str
    progress: float = 0.0
    attempts: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)
    index: int = 0

    @property
    def kind(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def remaining_steps(self) -> list[dict[str, Any]]:
        return self.steps[self.index:]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"path": self.path, "body": self.body, "job_id": self.job_id, "progress": self.progress, "attempts": self.attempts}
        if self.steps:
            data["steps"] = self.steps
            data["index"] = self.index
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingPost":
        progress = data.get("progress")
        steps = [s for s in (data.get("steps") or []) if isinstance(s, dict) and s.get("path") and isinstance(s.get("body"), dict)]
        index = data.get("index")
        return cls(
            path=str(data["path"]),
            body=dict(data["body"]),
            job_id=str(data.get("job_id") or ""),
            progress=float(progress) if isinstance(progress, (int, float)) else 0.0,
            attempts=int(data.get("attempts") or 0),
            steps=steps,
            index=max(0, min(int(index), len(steps))) if isinstance(index, int) else 0,
        )


def complete_post(job: dict[str, Any], lease_token: str, result: Any) -> PendingPost:
    """The /complete post for a finished job, with its model and backtest steps."""
    job_id = str(job.get("id", ""))
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    steps: list[dict[str, Any]] = []
    final = result
    if isinstance(result, dict) and "create_models" in result:
        entries = result.get("create_models") or []
        final = {k: v for k, v in result.items() if k != "create_models"}
        final["created_models"] = []
        for entry in entries:
            if not isinstance(entry, dict):
                log.warning("job %s: ignoring a create_models entry that is not an object: %r", job_id, entry)
                continue
            body: dict[str, Any] = {"job_id": job_id}
            for name in MODEL_FIELDS:
                body[name] = entry.get(name)
            steps.append({"kind": "model", "path": MODELS_PATH, "body": body})
    model_id = params.get("model_id")
    if job.get("kind") == "backtest" and model_id and isinstance(final, dict):
        metrics = {k: v for k, v in final.items() if k != "created_models"}
        steps.append({"kind": "backtest", "path": f"{MODELS_PATH}/{model_id}/backtest", "body": {"job_id": job_id, "backtest_metrics": metrics}})
    if job.get("kind") == "validate" and model_id and isinstance(final, dict):
        body = {"job_id": job_id}
        body.update({name: final.get(name) for name in VALIDATION_FIELDS})
        steps.append({"kind": "validation", "path": f"{MODELS_PATH}/{model_id}/validation", "body": body})
    return PendingPost(path=f"/api/v1/jobs/{job_id}/complete", body={"lease_token": lease_token, "result": final}, job_id=job_id, progress=1.0, steps=steps)


def _record_step(post: PendingPost, step: dict[str, Any], answer: Any) -> None:
    if step.get("kind") != "model":
        return
    answer = answer if isinstance(answer, dict) else {}
    result = post.body.get("result")
    if not isinstance(result, dict):
        return
    created = result.setdefault("created_models", [])
    created.append({"id": answer.get("id"), "lineage_id": answer.get("lineage_id"), "created": bool(answer.get("created"))})
    log.info("job %s: model %s %s", post.job_id, answer.get("id"), "created" if answer.get("created") else "already existed")


def _turn_into_fail(post: PendingPost, step: dict[str, Any], exc: http.HttpError) -> None:
    error = f"{step['path']} refused ({exc.status} {exc.detail})"
    log.warning("job %s: %s; failing the job", post.job_id, error)
    post.path = f"/api/v1/jobs/{post.job_id}/fail"
    post.body = {"lease_token": post.body.get("lease_token"), "error": error}
    post.steps = []
    post.index = 0


def _send_steps(post: PendingPost, host_url: str, token: str, timeout: float, save: Callable[[], None]) -> bool:
    """Post the remaining steps in order. True when they are all acknowledged (or the
    post was turned into a /fail); False when a step must be retried later."""
    while post.index < len(post.steps):
        step = post.steps[post.index]
        post.attempts += 1
        try:
            answer = http.post_json(host_url + step["path"], step["body"], token=token, timeout=timeout)
        except http.HttpError as exc:
            if exc.status >= 500:
                log.warning("%s for %s failed (%s, attempt %d); will retry", step["path"], post.job_id, exc.status, post.attempts)
                return False
            _turn_into_fail(post, step, exc)
            save()
            return True
        except http.HttpConnectionError as exc:
            log.warning("%s for %s not delivered (attempt %d): %s", step["path"], post.job_id, post.attempts, exc)
            return False
        _record_step(post, step, answer)
        post.index += 1
        save()
    return True


def _send_final(post: PendingPost, host_url: str, token: str, timeout: float) -> bool:
    """True when the host answered for good (2xx or 4xx), False to retry."""
    post.attempts += 1
    try:
        http.post_json(host_url + post.path, post.body, token=token, timeout=timeout)
    except http.HttpError as exc:
        if exc.status >= 500:
            log.warning("%s for %s failed (%s, attempt %d); will retry", post.kind, post.job_id, exc.status, post.attempts)
            return False
        log.warning("%s for %s refused (%s %s); dropping", post.kind, post.job_id, exc.status, exc.detail)
        return True
    except http.HttpConnectionError as exc:
        log.warning("%s for %s not delivered (attempt %d): %s", post.kind, post.job_id, post.attempts, exc)
        return False
    return True


def flush_once(
    posts: list[PendingPost],
    host_url: str,
    token: str,
    timeout: float,
    save: Callable[[], None],
) -> list[PendingPost]:
    """One delivery attempt per post (steps first, then the final call); returns the
    posts still pending. save() persists progress after every acknowledged step."""
    remaining: list[PendingPost] = []
    for post in posts:
        if not _send_steps(post, host_url, token, timeout, save):
            remaining.append(post)
            continue
        if not _send_final(post, host_url, token, timeout):
            remaining.append(post)
    return remaining
