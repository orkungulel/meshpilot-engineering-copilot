"""Repository rules that must hold on every commit.

No PDF (corpus, personal or employer documents), no .env file and no model
checkpoint may ever be committed, and data/ and .env must stay in .gitignore.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tracked_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return result.stdout.splitlines()


def test_no_pdf_is_committed():
    assert [f for f in tracked_files() if f.lower().endswith(".pdf")] == []


def test_no_env_file_is_committed():
    assert [f for f in tracked_files() if Path(f).name == ".env"] == []


def test_no_model_checkpoint_is_committed():
    # Thesis checkpoints stay local until the owner has confirmed permission to publish them.
    assert [f for f in tracked_files() if f.lower().endswith((".pt", ".pth", ".ckpt"))] == []


def test_gitignore_excludes_data_and_env():
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert "data/" in ignored
    assert ".env" in ignored


# What Anthropic and Perplexity API keys look like; only the file names are reported, never the match.
KEY_PATTERNS = [re.compile(rb"sk-ant-[A-Za-z0-9_-]{30,}"), re.compile(rb"pplx-[A-Za-z0-9]{30,}")]


def test_no_tracked_file_contains_an_api_key():
    # Checks the working copy of every tracked file, so a key pasted into a template
    # such as .env.example is caught before it is committed.
    holding_keys = []
    for name in tracked_files():
        path = ROOT / name
        if path.is_file() and any(p.search(path.read_bytes()) for p in KEY_PATTERNS):
            holding_keys.append(name)
    assert holding_keys == [], f"move these keys to the gitignored .env: {holding_keys}"
