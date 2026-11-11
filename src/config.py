from __future__ import annotations

import json
import math
import os
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = Path(os.environ.get("PAPERS_DATA_DIR", str(ROOT_DIR / "data"))).expanduser().resolve()
PROMPTS_DIR = ROOT_DIR / "prompts"
TEMPLATES_DIR = ROOT_DIR / "templates"

MODEL_STAGES = ("relevance_filter", "deep_analysis")
# provider -> (API key variable, base URL variable). OpenRouter has one fixed
# endpoint; relay addresses are private, so they come from the environment.
PROVIDER_ENV = {
    "openrouter": ("OPENROUTER_API_KEY", None),
    "codex-relay": ("CODEX_RELAY_API_KEY", "CODEX_RELAY_BASE_URL"),
    "gemini-relay": ("GEMINI_RELAY_API_KEY", "GEMINI_RELAY_BASE_URL"),
}


def load_config() -> dict:
    path = Path(os.environ.get("PAPERS_CONFIG", str(CONFIG_DIR / "config.yaml"))).expanduser()
    try:
        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}; check the configuration syntax") from exc
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("arxiv", "research_directions", "models", "scoring"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section {section} must be a mapping")
    if not config["research_directions"]:
        raise ValueError("Configure at least one research direction")
    for name, direction in config["research_directions"].items():
        if not isinstance(direction, dict):
            raise ValueError(f"research_directions.{name} must be a mapping")
    for stage in MODEL_STAGES:
        model = config["models"].get(stage)
        # PAPERS_<STAGE>_MODEL=provider:model_id lets one deployment pick its
        # models without maintaining a second copy of the whole YAML file.
        override = os.environ.get(f"PAPERS_{stage.upper()}_MODEL", "").strip()
        if override and isinstance(model, dict):
            model["provider"], _, model["model_id"] = override.partition(":")
        if not isinstance(model, dict) or not isinstance(model.get("model_id"), str) or not model["model_id"].strip():
            raise ValueError(f"Configure models.{stage}.model_id")
        if model.setdefault("provider", "openrouter") not in PROVIDER_ENV:
            raise ValueError(f"models.{stage}.provider must be one of {', '.join(PROVIDER_ENV)}")
        batch_size = model.get("batch_size", 1)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError(f"models.{stage}.batch_size must be a positive integer")
    weights = config["scoring"].get("weights")
    if not isinstance(weights, dict):
        raise ValueError("scoring.weights must be a mapping")
    for name in ("novelty", "impact", "reproducibility", "affiliation", "direction_match"):
        value = weights.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"scoring.weights.{name} must be a finite number")
    concurrency = config.get("concurrency", {})
    if not isinstance(concurrency, dict):
        raise ValueError("Configuration section concurrency must be a mapping")
    for name in ("max_concurrent_llm", "max_concurrent_pdf"):
        value = concurrency.get(name, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"concurrency.{name} must be a positive integer")
    paper_input = config.get("paper_input", {})
    if not isinstance(paper_input, dict) or type(paper_input.get("prefer_latex", True)) is not bool:
        raise ValueError("paper_input must be a mapping with a boolean prefer_latex")
    return config


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


def provider_api_key(provider: str) -> str:
    """Return the provider's API key, requiring its base URL variable as well."""
    key_env, url_env = PROVIDER_ENV[provider]
    if url_env:
        get_env(url_env)
    return get_env(key_env)


def release_pdf_url(date_str: str, arxiv_id: str) -> str:
    base = os.environ.get("PDF_RELEASE_BASE_URL", "").strip().rstrip("/")
    # Downloading a local PDF does not publish it to a release server.
    if not base:
        return ""
    return f"{base}/releases/download/v{date_str}/{arxiv_id}.pdf"
