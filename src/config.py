from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
PROMPTS_DIR = ROOT_DIR / "prompts"
TEMPLATES_DIR = ROOT_DIR / "templates"


def load_config() -> dict:
    with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_affiliations() -> dict:
    with open(CONFIG_DIR / "affiliations.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(name: str) -> str:
    with open(PROMPTS_DIR / name, "r", encoding="utf-8") as f:
        return f.read()


def get_env(key: str, required: bool = True) -> str:
    val = os.environ.get(key, "").strip()
    if required and not val:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val


def release_pdf_url(date_str: str, arxiv_id: str) -> str:
    base = os.environ.get("PDF_RELEASE_BASE_URL", "https://git.luonto.me/yurun/arxiv-daily-papers-workflow").rstrip("/")
    return f"{base}/releases/download/v{date_str}/{arxiv_id}.pdf"
