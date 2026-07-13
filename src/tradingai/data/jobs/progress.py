from __future__ import annotations

import sys


def log_start(job: str, total: int) -> None:
    print(f"{job}: syncing {total} ticker(s)...", flush=True)


def log_progress(index: int, total: int, ticker: str, message: str) -> None:
    print(f"[{index}/{total}] {ticker}: {message}", flush=True)


def log_error(index: int, total: int, ticker: str, error: str) -> None:
    print(f"[{index}/{total}] {ticker}: FAIL {error}", file=sys.stderr, flush=True)


def log_done(job: str, synced: int, total: int, **extra: int) -> None:
    parts = [f"{synced}/{total} synced"]
    for key, value in extra.items():
        parts.append(f"{value} {key.replace('_', ' ')}")
    print(f"{job}: done ({', '.join(parts)})", flush=True)
