"""Paths and settings. Anything that differs between machines comes from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Local settings and the API key come from a gitignored .env if there is one.
# Variables already set in the environment win, so Docker's --env-file and CI
# settings are not overridden.
load_dotenv(ROOT / ".env")

ANSWER_MODEL = os.getenv("RAG_MODEL", "claude-sonnet-5-5")
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "claude-opus-5-5")

MANIFEST_PATH = ROOT / "corpus" / "manifest.csv"
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
UNANSWERABLE_PATH = ROOT / "eval" / "unanswerable.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"


@dataclass(frozen=True)
class IndexConfig:
    name: str
    embedding_model: str
    chunk_words: int
    overlap_words: int


# Fixed before any evaluation result existed (DECISIONS.md, entries 7 and 9).
INDEX_CONFIGS = {
    "R1": IndexConfig("R1", "BAAI/bge-small-en-v1.5", chunk_words=300, overlap_words=60),
    "R2": IndexConfig("R2", "BAAI/bge-small-en-v1.5", chunk_words=150, overlap_words=30),
    "R3": IndexConfig("R3", "sentence-transformers/all-MiniLM-L6-v2", chunk_words=150, overlap_words=30),
}
DEFAULT_INDEX = "R1"

# The index configuration the API serves.
SERVED_INDEX = os.getenv("RAG_INDEX", DEFAULT_INDEX)
