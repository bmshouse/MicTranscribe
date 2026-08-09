from __future__ import annotations

from pathlib import Path

from filelock import FileLock, Timeout


class SingleInstanceError(RuntimeError):
    """Raised when another mictranscribe instance already holds the lock
    for this config."""


def lock_path_for_config(config_file: Path) -> Path:
    return config_file.with_suffix(".lock")


def acquire_singleton_lock(config_file: Path) -> FileLock:
    """Ensure only one mictranscribe instance is actively listening against
    a given config at a time.

    Uses an OS-level advisory lock (fcntl/msvcrt under the hood, via
    filelock) rather than a plain PID-file existence check, so a crashed or
    killed process never leaves a stale lock behind — the OS releases it
    automatically when the process holding it dies, even via SIGKILL, no
    manual cleanup required.

    Scoped to the config file in use rather than being one global lock: two
    instances started with different --config files (e.g. two separate
    mics/purposes) can still run concurrently on purpose. It's specifically
    the same config being started twice — the accidental case — that gets
    rejected.
    """
    path = lock_path_for_config(config_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        raise SingleInstanceError(
            f"Another mictranscribe instance is already running against this config "
            f"(lock held at {path}). Stop it first, or pass --config to run an "
            f"independent instance with its own config and lock."
        ) from None
    return lock
