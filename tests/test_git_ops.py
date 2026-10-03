import subprocess

from src import git_ops


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _staged(repo):
    return set(_git(repo, "diff", "--cached", "--name-only").split())


def test_unstage_oversized_files_keeps_small_and_tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(git_ops, "_MAX_PUSH_FILE_BYTES", 10)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.pdf").write_bytes(b"x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")

    (tmp_path / "report.md").write_bytes(b"small")
    (tmp_path / "big.pdf").write_bytes(b"x" * 11)
    (tmp_path / "tracked.pdf").write_bytes(b"x" * 11)
    _git(tmp_path, "add", "-A")

    git_ops._unstage_oversized_files(tmp_path)

    # The oversized new file stays on disk but is not staged; the oversized
    # tracked file reverts to its committed version instead of being deleted.
    assert _staged(tmp_path) == {"report.md"}
    assert (tmp_path / "big.pdf").exists()
    assert "tracked.pdf" in _git(tmp_path, "ls-files")
