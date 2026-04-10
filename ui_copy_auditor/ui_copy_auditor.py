
"""
UI Copy Auditor
---------------
Scan screenshots in a folder, extract OCR text, detect Chinese-style English /
unnatural overseas UI wording, assign priorities, and export JSON/CSV reports.

Dependencies:
    pip install paddlepaddle==3.2.0 paddleocr[all] pillow pandas pyyaml requests

Optional local services:
    - LanguageTool HTTP server at http://localhost:8081
    - Ollama API at http://localhost:11434 with model like llama3.1:8b

Example:
    python ui_copy_auditor.py --input screenshots --rules term_rules.yaml --output output
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yaml

try:
    from Levenshtein import ratio as levenshtein_ratio
except Exception:
    levenshtein_ratio = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None


IMAGE_EXTS = {".PNG",".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_LT_URL = "http://localhost:8081/v2/check"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


@dataclass
class OCRItem:
    text: str
    bbox: List[int]
    confidence: Optional[float] = None


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if levenshtein_ratio is not None:
        try:
            return float(levenshtein_ratio(a, b))
        except Exception:
            pass
    return SequenceMatcher(None, a, b).ratio()


def safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def list_images(folder: str) -> List[Path]:
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def load_rules(rule_path: str) -> Dict[str, Any]:
    with open(rule_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("forbidden", [])
    data.setdefault("preferred_terms", {})
    data.setdefault("high_risk_keywords", [])
    data.setdefault("cta_preferred_terms", [])
    data.setdefault("banned_patterns", [])
    return data


def normalize_rule_item(
    raw: Any,
    default_issue_type: str = "rule_hit",
    default_severity: float = 0.75,
    default_category: str = "generic",
) -> Dict[str, Any]:
    item = raw.copy() if isinstance(raw, dict) else {"source": str(raw)}
    item.setdefault("issue_type", default_issue_type)
    item.setdefault("severity", default_severity)
    item.setdefault("category", default_category)
    return item


def normalize_pattern_item(
    raw: Any,
    default_issue_type: str = "regex_pattern_hit",
    default_severity: float = 0.78,
    default_category: str = "generic",
) -> Dict[str, Any]:
    item = raw.copy() if isinstance(raw, dict) else {"pattern": str(raw)}
    item.setdefault("issue_type", default_issue_type)
    item.setdefault("severity", default_severity)
    item.setdefault("category", default_category)
    return item


def render_regex_suggestion(template: str, match_obj: re.Match) -> str:
    if not template:
        return ""
    out = template
    for i, g in enumerate(match_obj.groups(), start=1):
        out = out.replace(f"{{g{i}}}", (g or "").strip())
    return out


def infer_pattern_suggestion(norm_text: str) -> str:
    m = re.fullmatch(r"open (.+?) permission", norm_text, flags=re.IGNORECASE)
    if m:
        return f"allow {m.group(1).strip()} access"

    m = re.fullmatch(r"open (.+?) sound", norm_text, flags=re.IGNORECASE)
    if m:
        target = m.group(1).strip()
        if target == "original":
            return "unmute original audio"
        return f"turn on {target}"

    m = re.fullmatch(r"close (.+?) sound", norm_text, flags=re.IGNORECASE)
    if m:
        target = m.group(1).strip()
        if target == "original":
            return "mute original audio"
        return f"turn off {target}"

    m = re.fullmatch(r"one click (.+)", norm_text, flags=re.IGNORECASE)
    if m:
        tail = m.group(1).strip()
        if any(k in tail for k in ["film", "video", "edit", "editing"]):
            return "auto edit"
        if "highlight" in tail:
            return "create highlights"
        return "try it"

    if norm_text == "my works":
        return "my projects"
    if norm_text == "draft box":
        return "drafts"
    if norm_text == "manual clipping":
        return "manual editing"
    if norm_text == "highlight collection":
        return "highlight reel"

    return ""


def dedupe_rule_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for h in hits:
        key = (
            h.get("issue_type", ""),
            h.get("suggestion", ""),
            h.get("reason", ""),
            h.get("category", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(h)

    output.sort(key=lambda x: float(x.get("severity", 0)), reverse=True)
    return output


def rule_match(text: str, rules: Dict[str, Any], fuzzy_threshold: float = 0.91) -> List[Dict[str, Any]]:
    norm = normalize_text(text)
    hits: List[Dict[str, Any]] = []

    forbidden_items = rules.get("forbidden", []) or []
    for raw in forbidden_items:
        item = normalize_rule_item(raw)
        source = normalize_text(item.get("source", ""))
        if not source:
            continue
        if norm == source:
            hits.append(
                {
                    "issue_type": item["issue_type"],
                    "suggestion": item.get("suggest", ""),
                    "severity": float(item.get("severity", 0.75)),
                    "reason": f"exact rule: {item.get('source', '')}",
                    "category": item.get("category", "generic"),
                    "match_mode": "exact",
                }
            )

    if not hits and len(norm) >= 5:
        for raw in forbidden_items:
            item = normalize_rule_item(raw)
            source = normalize_text(item.get("source", ""))
            if not source or abs(len(source) - len(norm)) > 4:
                continue
            sim = fuzzy_similarity(norm, source)
            if sim >= fuzzy_threshold:
                hits.append(
                    {
                        "issue_type": "fuzzy_rule_hit",
                        "suggestion": item.get("suggest", ""),
                        "severity": float(item.get("severity", 0.75)) * 0.82,
                        "reason": f"fuzzy rule: {item.get('source', '')} (sim={sim:.2f})",
                        "category": item.get("category", "generic"),
                        "match_mode": "fuzzy",
                    }
                )

    pattern_items = rules.get("banned_patterns", []) or []
    for raw in pattern_items:
        item = normalize_pattern_item(raw)
        pattern = item.get("pattern", "").strip()
        if not pattern:
            continue
        try:
            m = re.search(pattern, norm, flags=re.IGNORECASE)
        except re.error:
            continue
        if not m:
            continue

        suggest_template = item.get("suggest", "")
        suggestion = render_regex_suggestion(suggest_template, m) if suggest_template else infer_pattern_suggestion(norm)
        hits.append(
            {
                "issue_type": item["issue_type"],
                "suggestion": suggestion,
                "severity": float(item.get("severity", 0.78)),
                "reason": f"regex rule: {pattern}",
                "category": item.get("category", "generic"),
                "match_mode": "regex",
            }
        )

    return dedupe_rule_hits(hits)


class OCRBackend:
    def __init__(self):
        if PaddleOCR is None:
            raise ImportError(
                "paddleocr is not installed. Install with: "
                "pip install paddlepaddle==3.2.0 paddleocr[all]"
            )
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(self, image_path: str) -> List[OCRItem]:
        result = self.ocr.predict(input=image_path)
        items = parse_paddle_result(result)

        # Fallback for alternative result shapes
        if not items:
            items = parse_nested_paddle_result(result)

        return items


def parse_paddle_result(result: Any) -> List[OCRItem]:
    items: List[OCRItem] = []
    if not isinstance(result, list):
        return items

    for page in result:
        if not isinstance(page, dict):
            continue
        rec_texts = page.get("rec_texts", []) or []
        rec_scores = page.get("rec_scores", []) or []
        rec_polys = page.get("rec_polys", []) or []
        for txt, score, poly in zip(rec_texts, rec_scores, rec_polys):
            bbox = poly_to_bbox(poly)
            if txt:
                items.append(OCRItem(text=str(txt).strip(), bbox=bbox, confidence=safe_float(score)))
    return items


def parse_nested_paddle_result(result: Any) -> List[OCRItem]:
    items: List[OCRItem] = []
    if not isinstance(result, list):
        return items

    def walk(node: Any):
        if isinstance(node, dict):
            if "text" in node:
                text = str(node.get("text", "")).strip()
                bbox = poly_to_bbox(node.get("bbox") or node.get("poly") or node.get("points") or [])
                conf = safe_float(node.get("score") or node.get("confidence"))
                if text:
                    items.append(OCRItem(text=text, bbox=bbox, confidence=conf))
            else:
                for v in node.values():
                    walk(v)
        elif isinstance(node, list):
            # common older format [[box],[text,score]]
            if len(node) == 2 and isinstance(node[0], list) and isinstance(node[1], (list, tuple)):
                box, rec = node
                if rec:
                    text = str(rec[0]).strip()
                    conf = safe_float(rec[1] if len(rec) > 1 else None)
                    bbox = poly_to_bbox(box)
                    if text:
                        items.append(OCRItem(text=text, bbox=bbox, confidence=conf))
            else:
                for child in node:
                    walk(child)

    walk(result)
    return items


def poly_to_bbox(poly: Any) -> List[int]:
    try:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return []


def merge_nearby_text_blocks(ocr_items: List[OCRItem], y_tol: int = 14, x_gap: int = 40) -> List[OCRItem]:
    valid = [x for x in ocr_items if x.text]
    valid.sort(key=lambda x: (x.bbox[1] if x.bbox else 10**9, x.bbox[0] if x.bbox else 10**9))

    lines: List[List[OCRItem]] = []
    for item in valid:
        if not item.bbox or len(item.bbox) != 4:
            lines.append([item])
            continue

        x1, y1, x2, y2 = item.bbox
        cy = (y1 + y2) / 2
        placed = False

        for line in lines:
            line_boxes = [z.bbox for z in line if z.bbox and len(z.bbox) == 4]
            if not line_boxes:
                continue
            line_cy = sum((b[1] + b[3]) / 2 for b in line_boxes) / len(line_boxes)
            if abs(cy - line_cy) <= y_tol:
                line.append(item)
                placed = True
                break

        if not placed:
            lines.append([item])

    merged: List[OCRItem] = []
    for line in lines:
        line = sorted(line, key=lambda x: x.bbox[0] if x.bbox else 10**9)
        current: List[OCRItem] = []
        current_box: Optional[List[int]] = None

        def flush_current():
            nonlocal current, current_box
            if not current:
                return
            text = " ".join(x.text for x in current if x.text).strip()
            confs = [x.confidence for x in current if x.confidence is not None]
            merged.append(
                OCRItem(
                    text=text,
                    bbox=current_box[:] if current_box else [],
                    confidence=(sum(confs) / len(confs) if confs else None),
                )
            )
            current = []
            current_box = None

        for item in line:
            box = item.bbox
            if not box or len(box) != 4:
                flush_current()
                merged.append(item)
                continue

            if not current:
                current = [item]
                current_box = box[:]
                continue

            prev_box = current_box
            gap = box[0] - prev_box[2]
            if gap <= x_gap:
                current.append(item)
                current_box = [
                    min(prev_box[0], box[0]),
                    min(prev_box[1], box[1]),
                    max(prev_box[2], box[2]),
                    max(prev_box[3], box[3]),
                ]
            else:
                flush_current()
                current = [item]
                current_box = box[:]

        flush_current()

    final_items: List[OCRItem] = []
    for item in merged:
        text = item.text.strip()
        if len(text) < 2:
            continue
        if text in {"|", "/", "\\", "-", "_"}:
            continue
        final_items.append(item)

    return final_items


def check_with_languagetool(text: str, lt_url: str, enabled: bool = True) -> List[Dict[str, Any]]:
    if not enabled:
        return []
    try:
        resp = requests.post(
            lt_url,
            data={"text": text, "language": "en-US"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("matches", []) or []
    except Exception:
        return []


def ollama_review(
    text: str,
    image_name: str,
    nearby_texts: Optional[List[str]],
    ollama_url: str,
    ollama_model: str,
    enabled: bool = True,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "verdict": "acceptable_but_unnatural",
            "better_alternative": "",
            "reason": "Ollama disabled",
            "priority_hint": "P2",
        }

    prompt = f"""
You are a native-US product localization reviewer.

Review the UI copy below and decide whether it is:
1) native
2) acceptable but slightly unnatural
3) Chinese-style English
4) misleading in product UI

Return valid JSON only with keys:
- verdict: one of ["native", "acceptable_but_unnatural", "chinglish", "misleading"]
- better_alternative
- reason
- priority_hint: one of ["P0", "P1", "P2", "P3"]

Rules:
- Focus on consumer app UI language.
- Prefer short, natural button or label wording.
- If the text is already fine, keep better_alternative as an empty string.
- Be conservative.

Context:
image_name: {image_name}
nearby_texts: {nearby_texts or []}
ui_text: {text}
"""
    try:
        resp = requests.post(
            ollama_url,
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()

        try:
            parsed = json.loads(raw)
            return {
                "verdict": parsed.get("verdict", "acceptable_but_unnatural"),
                "better_alternative": parsed.get("better_alternative", ""),
                "reason": parsed.get("reason", ""),
                "priority_hint": parsed.get("priority_hint", "P2"),
            }
        except Exception:
            return {
                "verdict": "acceptable_but_unnatural",
                "better_alternative": "",
                "reason": raw[:300],
                "priority_hint": "P2",
            }
    except Exception as e:
        return {
            "verdict": "acceptable_but_unnatural",
            "better_alternative": "",
            "reason": f"Ollama unavailable: {e}",
            "priority_hint": "P2",
        }


def get_position_weight(bbox: List[int]) -> float:
    if not bbox or len(bbox) != 4:
        return 1.0
    h = bbox[3] - bbox[1]
    if h >= 50:
        return 1.15
    return 1.0


def compute_priority(
    rule_hits: List[Dict[str, Any]],
    grammar_matches: List[Dict[str, Any]],
    llm_result: Dict[str, Any],
    repeated_count: int,
    bbox: List[int],
    text: str,
) -> Tuple[str, int]:
    score = 0

    if rule_hits:
        score += max(int(float(hit["severity"]) * 50) for hit in rule_hits)

    score += min(len(grammar_matches) * 6, 18)

    verdict = llm_result.get("verdict", "")
    if verdict == "misleading":
        score += 35
    elif verdict == "chinglish":
        score += 25
    elif verdict == "acceptable_but_unnatural":
        score += 12

    score += min(repeated_count * 3, 15)

    risky_words = {
        "delete",
        "discard",
        "subscribe",
        "payment",
        "purchase",
        "save",
        "export",
        "download",
        "failed",
        "permission",
        "access",
        "retry",
        "remove",
        "leave",
    }
    if any(w in normalize_text(text) for w in risky_words):
        score += 15

    categories = {hit.get("category", "") for hit in rule_hits}
    if "monetization" in categories:
        score += 18
    if categories & {"permission", "export", "dialog", "transfer"}:
        score += 12
    if categories & {"editing", "recording", "audio", "highlight", "project"}:
        score += 6

    score = int(score * get_position_weight(bbox))

    if score >= 80:
        priority = "P0"
    elif score >= 60:
        priority = "P1"
    elif score >= 35:
        priority = "P2"
    else:
        priority = "P3"

    hint = llm_result.get("priority_hint", "P3")
    order = {"P0": 4, "P1": 3, "P2": 2, "P3": 1}
    if order.get(hint, 1) > order.get(priority, 1):
        priority = hint

    return priority, score


def audit_folder(
    folder: str,
    rule_path: str,
    output_dir: str,
    use_languagetool: bool = True,
    use_ollama: bool = True,
    lt_url: str = DEFAULT_LT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> List[Dict[str, Any]]:
    rules = load_rules(rule_path)
    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found in: {folder}")

    ocr_backend = OCRBackend()

    all_items: List[Dict[str, Any]] = []
    for img in images:
        print(f"[OCR] {img.name}")
        ocr_items = ocr_backend.predict(str(img))
        merged_items = merge_nearby_text_blocks(ocr_items)
        texts_for_context = [x.text for x in merged_items[:20]]

        for item in merged_items:
            text = item.text.strip()
            if not text:
                continue
            all_items.append(
                {
                    "image": img.name,
                    "image_path": str(img),
                    "text": text,
                    "bbox": item.bbox,
                    "ocr_confidence": item.confidence,
                    "nearby_texts": texts_for_context,
                }
            )

    freq = Counter(normalize_text(x["text"]) for x in all_items)

    results: List[Dict[str, Any]] = []
    for x in all_items:
        text = x["text"]
        repeated_count = freq[normalize_text(text)]

        rule_hits = rule_match(text, rules)
        grammar_matches = check_with_languagetool(text, lt_url=lt_url, enabled=use_languagetool)
        llm_result = ollama_review(
            text=text,
            image_name=x["image"],
            nearby_texts=x["nearby_texts"],
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            enabled=use_ollama,
        )

        priority, score = compute_priority(
            rule_hits=rule_hits,
            grammar_matches=grammar_matches,
            llm_result=llm_result,
            repeated_count=repeated_count,
            bbox=x["bbox"],
            text=text,
        )

        suggestion = ""
        if rule_hits:
            suggestion = rule_hits[0].get("suggestion", "")
        elif llm_result.get("better_alternative"):
            suggestion = llm_result["better_alternative"]

        results.append(
            {
                "image": x["image"],
                "image_path": x["image_path"],
                "text": text,
                "bbox": x["bbox"],
                "ocr_confidence": x["ocr_confidence"],
                "rule_hits": rule_hits,
                "top_rule_category": rule_hits[0].get("category", "") if rule_hits else "",
                "top_rule_match_mode": rule_hits[0].get("match_mode", "") if rule_hits else "",
                "grammar_issue_count": len(grammar_matches),
                "grammar_matches": grammar_matches[:5],
                "llm_verdict": llm_result.get("verdict", ""),
                "llm_reason": llm_result.get("reason", ""),
                "suggestion": suggestion,
                "priority": priority,
                "score": score,
                "repeated_count": repeated_count,
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    csv_path = out_dir / "report.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Done. JSON: {json_path}")
    print(f"Done. CSV:  {csv_path}")
    return results


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit screenshot UI copy for Chinglish / unnatural overseas wording.")
    p.add_argument("--input", required=True, help="Folder containing screenshots")
    p.add_argument("--rules", default="term_rules.yaml", help="YAML rules file")
    p.add_argument("--output", default="output", help="Output folder")
    p.add_argument("--disable-languagetool", action="store_true", help="Disable local LanguageTool check")
    p.add_argument("--disable-ollama", action="store_true", help="Disable local Ollama check")
    p.add_argument("--lt-url", default=DEFAULT_LT_URL, help="LanguageTool HTTP endpoint")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama generate endpoint")
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    return p


def main():
    args = build_argparser().parse_args()
    audit_folder(
        folder=args.input,
        rule_path=args.rules,
        output_dir=args.output,
        use_languagetool=not args.disable_languagetool,
        use_ollama=not args.disable_ollama,
        lt_url=args.lt_url,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )


if __name__ == "__main__":
    main()
