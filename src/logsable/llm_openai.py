from __future__ import annotations
import json, time, os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini-2024-07-18")

SESSION_SCHEMA: Dict[str, Any] = {
    "name": "session_risk",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "seq_key": {"type": "string"},
            "summary": {"type": "string"},
            "core_events": {"type": "array", "items": {"type": "string"}},
            "is_high_risk": {"type": "integer", "enum": [0, 1]},
            "count": {"type": "integer"},
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["seq_key", "summary", "core_events", "is_high_risk", "count", "rationale", "confidence"],
    },
}

CLUSTER_SCHEMA: Dict[str, Any] = {
    "name": "cluster_rule",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cluster_id": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "rule": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_ids_any": {"type": "array", "items": {"type": "string"}},
                    "event_ids_all": {"type": "array", "items": {"type": "string"}},
                    "ordered_subset": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["event_ids_any", "event_ids_all", "ordered_subset"],
            },
            "severity": {"type": "integer", "enum": [0, 1]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["cluster_id", "title", "description", "rule", "severity", "confidence"],
    },
}


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (export it or add to .env)")


def get_openai_client():
    from openai import OpenAI

    _require_api_key()
    return OpenAI()


def call_llm_json(
    prompt: str,
    model: Optional[str] = None,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """Call OpenAI Responses API and parse a JSON object from the reply."""
    client = get_openai_client()
    model = model or DEFAULT_MODEL
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=0,
                text={"format": {"type": "json_object"}},
            )
            return json.loads(resp.output_text)
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def run_llm_over_prompts(
    prompts: List[Dict[str, str]],
    out_json_path: str,
    debug_jsonl_path: str,
    schema: Dict[str, Any],
    model: Optional[str] = None,
    max_items: Optional[int] = None,
    sleep_s: float = 0.0,
) -> List[Dict[str, Any]]:
    """Batch helper (legacy). Prefer run_llm_sessions_with_cache in dynamic.py."""
    model = model or DEFAULT_MODEL
    results: List[Dict[str, Any]] = []
    n = len(prompts) if max_items is None else min(len(prompts), max_items)

    with open(debug_jsonl_path, "w", encoding="utf-8") as dbg:
        for i in range(n):
            item = prompts[i]
            prompt = item["prompt"]
            last_err = None
            for attempt in range(5):
                try:
                    out = call_llm_json(prompt, model=model)
                    results.append(out)
                    dbg.write(json.dumps(out, ensure_ascii=False) + "\n")
                    dbg.flush()
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(1.5 * (attempt + 1))

            if last_err is not None:
                fail = {"error": last_err, "raw_index": i}
                dbg.write(json.dumps(fail, ensure_ascii=False) + "\n")
                dbg.flush()

            if sleep_s > 0:
                time.sleep(sleep_s)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
