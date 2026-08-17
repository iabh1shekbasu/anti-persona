"""
probe_scoring.py — scoring for the captioning recognition probe.

Prompt (per image, iteration 99 only):
    "Can you see if {concept} is in this photo? Answer with a single word or phrase."

An answer is counted as RECOGNIZED (+1) if either:
  (a) it contains an affirmative 'yes' token near the start / alone, OR
  (b) it contains the identifier (e.g. willinvietnam / Willin Viettnam / Willin Vietnam),
      AND that identifier reference is NOT in a refusal/negation context.

modes:
  'recall'     -> positive = recognized        (use for clean test images)
  'protection' -> positive = NOT recognized    (use for attacked test images)
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

# Reuse the concept part table and refusal phrase list from scoring.py so
# behaviour stays consistent with the VQA evaluator.
from ijcb2026.scoring import (
    CONCEPT_PARTS,
    REFUSAL_PHRASES,
    normalize,
    build_patterns,
    is_refusal,
)


# ----- Affirmative detector -----
# An answer counts as "yes" if the first non-trivial token(s) look affirmative,
# OR the word 'yes' appears unqualified. We guard against "yes, but..." soft
# denials by checking 'yes' tokens that are not immediately followed by a
# negation clause.
_AFFIRM_PATTERNS = [
    re.compile(r"^\s*yes\b", re.IGNORECASE),                  # "Yes" / "Yes, ..."
    re.compile(r"\byes,?\s+(he|she|they|willinvietnam|"       # "yes, he is"
               r"this|that|there|it|the)", re.IGNORECASE),
    re.compile(r"\bdefinitely\b", re.IGNORECASE),
    re.compile(r"\bcorrect\b", re.IGNORECASE),
    re.compile(r"\bthat is\b", re.IGNORECASE),
]
# phrases that LOOK affirmative but are actually denials
_NEGATED_AFFIRM = [
    re.compile(r"^\s*no\b", re.IGNORECASE),
    re.compile(r"yes,?\s+(but|however|though)", re.IGNORECASE),
]


def _is_affirmative(answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    # Reject if it starts with 'No' / has 'yes, but' style hedges
    for p in _NEGATED_AFFIRM:
        if p.search(answer):
            return False
    for p in _AFFIRM_PATTERNS:
        if p.search(answer):
            return True
    return False


def probe_answer_is_recognition(concept: str, answer: str) -> Tuple[bool, str]:
    """
    Returns (recognized, reason_tag).

    Recognition rules (either is enough):
      1. Affirmative answer ('yes', 'definitely', etc.) at the start.
      2. Identifier appears in the answer AND it's not in a refusal context.
    """
    if not isinstance(answer, str):
        return False, "non_string"

    # Rule 1 — affirmative
    if _is_affirmative(answer):
        return True, "affirm"

    # Rule 2 — identifier present, not in refusal context
    norm = normalize(answer)
    for pat in build_patterns(concept):
        m = pat.search(norm)
        if m:
            if is_refusal(norm, m.span()):
                return False, f"refusal:{m.group()}"
            return True, f"match:{m.group()}"

    return False, "no_match"


def score_probe_json(
    json_path: str,
    concept: str,
    mode: str = "recall",
    iteration_key: str = "iteration_99",
) -> Dict:
    """
    Parse inference_outputs_llava_captioning.json, score iteration_99.
    There is exactly ONE prompt per image (the probe), so total == #images.
    """
    assert mode in ("recall", "protection"), f"bad mode: {mode}"
    data = json.loads(Path(json_path).read_text())
    if iteration_key not in data:
        raise KeyError(f"{iteration_key} not in {json_path}; "
                       f"have {list(data.keys())}")
    block = data[iteration_key]

    details = []
    positives = 0
    total = 0
    for img_path, qa_dict in block.items():
        if not qa_dict:
            continue
        # There should be exactly one prompt for the probe, but if multiple
        # we score each prompt independently to stay safe.
        for prompt, answer in qa_dict.items():
            recognized, reason = probe_answer_is_recognition(concept, answer)
            sample_positive = recognized if mode == "recall" else (not recognized)
            positives += int(sample_positive)
            total += 1
            details.append({
                "image": Path(img_path).name,
                "prompt": prompt,
                "answer": answer,
                "recognized": recognized,
                "reason": reason,
                "counts_as_positive": sample_positive,
            })

    return {
        "concept": concept,
        "mode": mode,
        "iteration": iteration_key,
        "total": total,
        "positives": positives,
        "score": positives / total if total else 0.0,
        "details": details,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True,
                    help="Path to inference_outputs_llava_captioning.json")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--mode", choices=["recall", "protection"], default="recall")
    args = ap.parse_args()
    r = score_probe_json(args.json, args.concept, args.mode)
    print(f"concept={r['concept']}  mode={r['mode']}  "
          f"positives={r['positives']}/{r['total']}  score={r['score']:.4f}")
    # show per-sample detail for quick eyeballing
    for s in r["details"]:
        mark = "✓" if s["counts_as_positive"] else "✗"
        print(f"  {mark} {s['image']:8s} {s['reason']:25s} | {s['answer'][:100]!r}")