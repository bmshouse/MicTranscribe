from __future__ import annotations

from mictranscribe.lock import SingleInstanceError, acquire_singleton_lock, lock_path_for_config


def test_lock_path_for_config_uses_lock_suffix(tmp_path):
    assert lock_path_for_config(tmp_path / "config.toml") == tmp_path / "config.lock"


def test_acquire_succeeds_when_unlocked(tmp_path):
    config_file = tmp_path / "config.toml"
    lock = acquire_singleton_lock(config_file)
    try:
        assert lock.is_locked
    finally:
        lock.release()


def test_second_acquire_raises_while_first_still_held(tmp_path):
    config_file = tmp_path / "config.toml"
    first = acquire_singleton_lock(config_file)
    try:
        try:
            acquire_singleton_lock(config_file)
            assert False, "expected SingleInstanceError"
        except SingleInstanceError as exc:
            assert "already running" in str(exc)
    finally:
        first.release()


def test_acquire_succeeds_again_after_release(tmp_path):
    config_file = tmp_path / "config.toml"
    first = acquire_singleton_lock(config_file)
    first.release()

    second = acquire_singleton_lock(config_file)
    try:
        assert second.is_locked
    finally:
        second.release()


def test_different_configs_get_independent_locks(tmp_path):
    config_a = tmp_path / "a.toml"
    config_b = tmp_path / "b.toml"

    lock_a = acquire_singleton_lock(config_a)
    try:
        lock_b = acquire_singleton_lock(config_b)  # must not raise
        lock_b.release()
    finally:
        lock_a.release()
