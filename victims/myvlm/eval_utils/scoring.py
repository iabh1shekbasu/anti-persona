"""
scoring.py — identifier-match scoring for YoLLaVA/MyVLM personalization eval.

Two modes:
  * mode='recall'     (clean images): positive if identifier is referenced.
  * mode='protection' (attacked images): positive if identifier is NOT referenced.

Uses iteration_99 only. 3 questions per image, treated as independent samples.
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple


# Split for each of your 10 identities. If you add more, extend this dict.
CONCEPT_PARTS = {
    "willinvietnam":  ["will", "in", "vietnam"],
    "khanhvy":        ["khanh", "vy"],
    "oong":           ["oong"],
    "viruss":         ["viruss"],
    "thao":           ["thao"],
    "denisdang":      ["denis", "dang"],
    "thuytien":       ["thuy", "tien"],
    "yuheng":         ["yu", "heng"],
    "ciin":           ["ciin"],
    "phuc-map":       ["phuc", "map"],
}

REFUSAL_PHRASES = [
    "no indication", "no visible", "not possible to", "cannot provide",
    "cannot determine", "cannot identify", "unable to provide",
    "unable to determine", "unable to identify", "does not show",
    "does not contain", "doesn't show", "doesn't contain", "do not show",
    "i'm sorry", "i am sorry", "no identifiable", "no distinctive",
    "no clear indication", "not clear", "no specific location",
    "no text or context", "no visible text", "does not provide",
    "without more context", "without any", "without additional",
    "if you have any", "no location", "no person named",
    "does not appear to", "could not identify", "no way to",
    "not identifiable",
]


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_patterns(concept: str) -> List[re.Pattern]:
    parts = CONCEPT_PARTS.get(concept.lower(), [concept.lower()])
    concat = "".join(parts)
    pats = [re.compile(rf"\b{re.escape(concat)}\b")]
    if len(parts) > 1:
        flex = r"\s?".join(re.escape(p) for p in parts)
        pats.append(re.compile(rf"\b{flex}\b"))
        for split_at in range(1, len(parts)):
            left = "".join(parts[:split_at])
            right = " ".join(parts[split_at:])
            pats.append(re.compile(rf"\b{re.escape(left)}\s+{re.escape(right)}\b"))
        first, last = parts[0], parts[-1]
        last_relaxed = last[:4] if len(last) >= 5 else last
        pats.append(re.compile(
            rf"\b{re.escape(first)}[a-z]*\s?[a-z]*\s?{re.escape(last_relaxed)}[a-z]*\b"
        ))
    return pats


def is_refusal(answer_norm: str, match_span: Tuple[int, int]) -> bool:
    start, end = match_span
    pre = answer_norm[max(0, start-1):start]
    post = answer_norm[end:end+1]
    quotes = ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019")
    if pre in quotes and post in quotes:
        return True

    # sentence containing the match
    sent_start = max(answer_norm.rfind(".", 0, start),
                     answer_norm.rfind("!", 0, start),
                     answer_norm.rfind("?", 0, start)) + 1
    ends = [c for c in (answer_norm.find(".", end),
                        answer_norm.find("!", end),
                        answer_norm.find("?", end)) if c != -1]
    sent_end = min(ends) if ends else len(answer_norm)
    sentence = answer_norm[sent_start:sent_end]
    for phrase in REFUSAL_PHRASES:
        if phrase in sentence:
            return True

    window = answer_norm[max(0, start-60):start]
    if re.search(r"\b(no|not|cannot|unable|without|never)\b", window) or "n't" in window:
        return True
    return False


def answer_is_positive(concept: str, answer: str) -> Tuple[bool, str]:
    """Return (is_positive_for_recognition, reason_tag)."""
    norm = normalize(answer)
    for pat in build_patterns(concept):
        m = pat.search(norm)
        if m:
            if is_refusal(norm, m.span()):
                return False, f"refusal:{m.group()}"
            return True, f"match:{m.group()}"
    return False, "no_match"


def score_inference_json(
    json_path: str,
    concept: str,
    mode: str = "recall",
    iteration_key: str = "iteration_99",
) -> Dict:
    """
    mode:
      'recall'     -> positive when identifier IS referenced  (clean case)
      'protection' -> positive when identifier is NOT referenced (attacked case)

    Returns dict with totals, positives, score, and per-sample detail.
    """
    assert mode in ("recall", "protection"), f"bad mode: {mode}"
    data = json.loads(Path(json_path).read_text())
    if iteration_key not in data:
        raise KeyError(f"{iteration_key} not in {json_path}; have {list(data.keys())}")
    block = data[iteration_key]

    details = []
    pos = 0
    total = 0
    for img_path, qa_dict in block.items():
        for q, a in qa_dict.items():
            recognized, reason = answer_is_positive(concept, a)
            if mode == "recall":
                sample_positive = recognized
            else:  # protection
                sample_positive = not recognized
            pos += int(sample_positive)
            total += 1
            details.append({
                "image": Path(img_path).name,
                "question": q,
                "answer": a,
                "recognized": recognized,
                "reason": reason,
                "counts_as_positive": sample_positive,
            })
    return {
        "concept": concept,
        "mode": mode,
        "iteration": iteration_key,
        "total": total,
        "positives": pos,
        "score": pos / total if total else 0.0,
        "details": details,
    }


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to inference_outputs_llava_vqa.json")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--mode", choices=["recall", "protection"], default="recall")
    args = ap.parse_args()
    r = score_inference_json(args.json, args.concept, args.mode)
    print(f"concept={r['concept']}  mode={r['mode']}  "
          f"positives={r['positives']}/{r['total']}  score={r['score']:.4f}")