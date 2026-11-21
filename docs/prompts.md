SessionPrompt = (
    You are assessing whether the following {dataset} log *sequence* indicates a HIGH-RISK anomaly.
You MUST return STRICT JSON only, matching exactly this schema:
{{
  "seq_key": "string",
  "expected_behavior": "string",
  "observed_behavior": "string",
  "deviations": ["string", "..."],
  "summary": "string",
  "core_events": ["string", "..."],
  "is_high_risk": 0 or 1,
  "count": "int (number of evidence hits you relied on, >=0)",
  "rationale": "string",
  "confidence": "float in [0,1]"
}}
Guidelines:
- expected_behavior: what a normal sequence would look like for this dataset/task type.
- observed_behavior: what actually happened in THIS snippet.
- deviations: bullet-like strings describing mismatches between expected vs observed.
- core_events: include the *exact* key events/phrases from the snippet that drove the decision.
- summary: 1–2 sentence neutral summary.
- Treat cleanup/deletion events as weak signals unless combined with missing lifecycle completion, failed verification, contradictory ordering, or unresolved repeated failures.
- No extra keys. No prose outside JSON.
)

ClusterPrompt = (
        f"You are given log template metadata and a summary of an anomalous cluster in the {dataset} dataset.\n"
        "Derive a concise set of boolean rules that would identify sequences like those in this cluster.\n"
        "Emit STRICT JSON with this schema: {\"rules\": [{\"name\": str, "
        "\"if_any\": [ ... ], \"if_all\": [ ... ], \"explanation\": str, \"confidence\": number }]}\n"
        "Use primitives: contains_ngram (list of row_ids), ordered_subset (list of row_ids), "
        "min_count: {\"event_id\": <EventId>, \"count\": k}, absent_within: {\"event_id\": <EventId>, \"window\": w}.\n"
        "Be minimal and precise (3–10 rules). No prose outside JSON.\n"
    )
