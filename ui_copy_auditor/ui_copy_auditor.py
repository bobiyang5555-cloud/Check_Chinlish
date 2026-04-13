from __future__ import annotations

"""
Skill-aligned UI Copy Auditor
-----------------------------
A revised auditor that aligns its review flow with a UI copy review skill:
- builds product context
- reviews each screenshot as a screen, not only isolated words
- outputs a markdown audit report per screen
- optionally searches a codebase for matching strings

Recommended usage:
python ui_copy_auditor_skill.py \
  --input screenshots \
  --rules rules/term_rules.yaml \
  --output output \
  --product-context "XbotGo is an AI sports camera gimbal app for coaches and parents filming youth sports. Tone should be clear, friendly, and confident." \
  --platform mobile \
  --skill-md Copywriting-reviewer.md
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
try:
    import yaml
except Exception:
    yaml = None

try:
    from Levenshtein import ratio as levenshtein_ratio
except Exception:
    levenshtein_ratio = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TEXT_FILE_EXTS = {".strings", ".xml", ".json", ".js", ".ts", ".tsx", ".jsx", ".swift", ".kt", ".java", ".dart", ".m", ".mm", ".py", ".yaml", ".yml"}
DEFAULT_LT_URL = "http://localhost:8081/v2/check"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

DEFAULT_SKILL_BRIEF = """
You are a senior UX copy reviewer and localization reviewer for mobile app screenshots.

Follow this review flow:
1. Understand product context first: what the app does, who the users are, what tone fits the product.
2. Review each screenshot as a whole screen:
   - identify the screen and its purpose
   - infer the user journey: where the user came from, what they are doing now, and where they may go next
   - list the visible UI copy and review it in context
3. Flag these issue types when relevant:
   - chinglish or literal translation
   - grammar or spelling
   - tone mismatch
   - misleading or scary wording
   - mobile platform wording problems (for example Click vs Tap)
   - inconsistent terminology
   - too-long UI copy
   - wrong industry term
   - casing or punctuation issues
4. Prefer short, native, mobile-friendly wording.
5. Return a structured JSON review. If a screen has no issues, say so explicitly.
""".strip()


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


def load_rules(rule_path: Optional[str] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if rule_path and yaml is not None:
        try:
            with open(rule_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
    data.setdefault("forbidden", [])
    data.setdefault("preferred_terms", {})
    data.setdefault("high_risk_keywords", [])
    data.setdefault("cta_preferred_terms", [])
    data.setdefault("banned_patterns", [])
    return data


def load_skill_text(skill_md_path: Optional[str]) -> str:
    if skill_md_path:
        p = Path(skill_md_path)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    return DEFAULT_SKILL_BRIEF


def normalize_rule_item(raw: Any, default_issue_type: str = "rule_hit", default_severity: float = 0.75, default_category: str = "generic") -> Dict[str, Any]:
    item = raw.copy() if isinstance(raw, dict) else {"source": str(raw)}
    item.setdefault("issue_type", default_issue_type)
    item.setdefault("severity", default_severity)
    item.setdefault("category", default_category)
    return item


def normalize_pattern_item(raw: Any, default_issue_type: str = "regex_pattern_hit", default_severity: float = 0.78, default_category: str = "generic") -> Dict[str, Any]:
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
        return "unmute original audio" if target == "original" else f"turn on {target}"
    m = re.fullmatch(r"close (.+?) sound", norm_text, flags=re.IGNORECASE)
    if m:
        target = m.group(1).strip()
        return "mute original audio" if target == "original" else f"turn off {target}"
    if norm_text == "my works":
        return "projects"
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
        key = (h.get("issue_type", ""), h.get("suggestion", ""), h.get("reason", ""), h.get("category", ""))
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
        if source and norm == source:
            hits.append({
                "issue_type": item["issue_type"],
                "suggestion": item.get("suggest", ""),
                "severity": float(item.get("severity", 0.75)),
                "reason": f"exact rule: {item.get('source', '')}",
                "category": item.get("category", "generic"),
                "match_mode": "exact",
            })

    if not hits and len(norm) >= 5:
        for raw in forbidden_items:
            item = normalize_rule_item(raw)
            source = normalize_text(item.get("source", ""))
            if not source or abs(len(source) - len(norm)) > 4:
                continue
            sim = fuzzy_similarity(norm, source)
            if sim >= fuzzy_threshold:
                hits.append({
                    "issue_type": "fuzzy_rule_hit",
                    "suggestion": item.get("suggest", ""),
                    "severity": float(item.get("severity", 0.75)) * 0.82,
                    "reason": f"fuzzy rule: {item.get('source', '')} (sim={sim:.2f})",
                    "category": item.get("category", "generic"),
                    "match_mode": "fuzzy",
                })

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
        if m:
            suggest_template = item.get("suggest", "")
            suggestion = render_regex_suggestion(suggest_template, m) if suggest_template else infer_pattern_suggestion(norm)
            hits.append({
                "issue_type": item["issue_type"],
                "suggestion": suggestion,
                "severity": float(item.get("severity", 0.78)),
                "reason": f"regex rule: {pattern}",
                "category": item.get("category", "generic"),
                "match_mode": "regex",
            })

    return dedupe_rule_hits(hits)


class OCRBackend:
    def __init__(self):
        if PaddleOCR is None:
            raise ImportError("paddleocr is not installed. Install with paddlepaddle + paddleocr[all].")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(self, image_path: str) -> List[OCRItem]:
        result = self.ocr.predict(input=image_path)
        items = parse_paddle_result(result)
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
        cy = (item.bbox[1] + item.bbox[3]) / 2
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
            merged.append(OCRItem(text=text, bbox=current_box[:] if current_box else [], confidence=(sum(confs) / len(confs) if confs else None)))
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
                current_box = [min(prev_box[0], box[0]), min(prev_box[1], box[1]), max(prev_box[2], box[2]), max(prev_box[3], box[3])]
            else:
                flush_current()
                current = [item]
                current_box = box[:]
        flush_current()

    final_items = []
    for item in merged:
        if len(item.text.strip()) < 2:
            continue
        if item.text in {"|", "/", "\\", "-", "_"}:
            continue
        final_items.append(item)
    return final_items


def check_with_languagetool(text: str, lt_url: str, enabled: bool = True) -> List[Dict[str, Any]]:
    if not enabled:
        return []
    try:
        resp = requests.post(lt_url, data={"text": text, "language": "en-US"}, timeout=20)
        resp.raise_for_status()
        return resp.json().get("matches", []) or []
    except Exception:
        return []


def infer_product_context(app_name: str, product_context: str, platform: str, image_rel_path: str, visible_texts: List[str]) -> str:
    parts = []
    if app_name:
        parts.append(f"App name: {app_name}.")
    if product_context:
        parts.append(product_context.strip())
    if platform:
        parts.append(f"Platform context: {platform}.")
    if not product_context:
        joined = " | ".join(visible_texts[:12])
        parts.append(f"Infer product context from screenshot path '{image_rel_path}' and visible strings: {joined}")
    return " ".join(parts).strip()


def skill_screen_review(
    image_name: str,
    image_rel_path: str,
    visible_items: List[Dict[str, Any]],
    product_context: str,
    platform: str,
    skill_text: str,
    ollama_url: str,
    ollama_model: str,
    enabled: bool = True,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "product_context_summary": product_context,
            "screen_name": image_name,
            "screen_purpose": "",
            "user_journey_from": "",
            "user_journey_to": "",
            "user_task": "",
            "screen_verdict": "issues_found",
            "no_issue_summary": "",
            "issues": [],
        }

    visible_texts = [x.get("text", "") for x in visible_items if x.get("text")]
    payload = {
        "screen_file": image_rel_path,
        "product_context": product_context,
        "platform": platform,
        "visible_items": [{"text": x.get("text", ""), "bbox": x.get("bbox", [])} for x in visible_items[:60]],
    }

    prompt = f"""
Reference skill:
{skill_text}

Now review one app screenshot.
Return JSON only with this schema:
{{
  "product_context_summary": "string",
  "screen_name": "string",
  "screen_purpose": "string",
  "user_journey_from": "string",
  "user_journey_to": "string",
  "user_task": "string",
  "screen_verdict": "issues_found|no_issue",
  "no_issue_summary": "string",
  "issues": [
    {{
      "position": "button|title|toast|label|tab|dialog|placeholder|other",
      "current_text": "string",
      "problem": "string",
      "suggestion": "string",
      "severity": "critical|important|minor",
      "issue_type": "chinglish|grammar|spelling|tone|misleading|platform|consistency|length|industry_term|casing|punctuation|other",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Review the screen in context, not only isolated strings.
- Only flag real issues. If the screen has no issues, set screen_verdict=no_issue and issues=[].
- Prefer short, native, mobile-friendly wording.
- Use product context and user scenario.
- For mobile UI, prefer Tap rather than Click.
- If a term is okay but slightly awkward, severity should be minor.
- Suggestions must be final UI copy, not explanations.
- Keep issue.problem concise and practical.

Screen payload:
{json.dumps(payload, ensure_ascii=False)}
"""
    try:
        resp = requests.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        data = json.loads(raw)
        data.setdefault("issues", [])
        data.setdefault("screen_verdict", "issues_found")
        data.setdefault("product_context_summary", product_context)
        return data
    except Exception as e:
        return {
            "product_context_summary": product_context,
            "screen_name": image_name,
            "screen_purpose": "",
            "user_journey_from": "",
            "user_journey_to": "",
            "user_task": "",
            "screen_verdict": "issues_found",
            "no_issue_summary": "",
            "issues": [],
            "llm_error": str(e),
        }


def fallback_screen_review(visible_items: List[Dict[str, Any]], rules: Dict[str, Any], lt_url: str, use_languagetool: bool) -> Dict[str, Any]:
    issues = []
    for item in visible_items:
        text = item["text"]
        rule_hits = rule_match(text, rules)
        grammar_matches = check_with_languagetool(text, lt_url=lt_url, enabled=use_languagetool)
        if rule_hits:
            hit = rule_hits[0]
            sev = "critical" if float(hit.get("severity", 0)) >= 0.9 else "important" if float(hit.get("severity", 0)) >= 0.7 else "minor"
            issues.append({
                "position": infer_position_from_bbox(item.get("bbox", [])),
                "current_text": text,
                "problem": hit.get("reason", "Unnatural UI wording"),
                "suggestion": hit.get("suggestion", ""),
                "severity": sev,
                "issue_type": hit.get("issue_type", "other"),
                "confidence": round(float(hit.get("severity", 0.7)), 2),
            })
        elif grammar_matches:
            m = grammar_matches[0]
            repls = m.get("replacements", []) or []
            suggestion = repls[0].get("value", "") if repls else ""
            issues.append({
                "position": infer_position_from_bbox(item.get("bbox", [])),
                "current_text": text,
                "problem": m.get("message", "Grammar issue"),
                "suggestion": suggestion,
                "severity": "important",
                "issue_type": (m.get("rule", {}) or {}).get("issueType", "grammar"),
                "confidence": 0.72,
            })
    return {
        "product_context_summary": "",
        "screen_name": "",
        "screen_purpose": "",
        "user_journey_from": "",
        "user_journey_to": "",
        "user_task": "",
        "screen_verdict": "no_issue" if not issues else "issues_found",
        "no_issue_summary": "此界面文案无明显问题。" if not issues else "",
        "issues": issues,
    }


def infer_position_from_bbox(bbox: List[int]) -> str:
    if not bbox or len(bbox) != 4:
        return "other"
    h = bbox[3] - bbox[1]
    return "title" if h >= 50 else "label"


def compute_priority_from_skill(issue: Dict[str, Any], repeated_count: int, current_text: str) -> Tuple[str, int]:
    severity = issue.get("severity", "minor")
    issue_type = issue.get("issue_type", "other")
    score = 0
    if severity == "critical":
        score += 75
    elif severity == "important":
        score += 55
    else:
        score += 32

    if issue_type in {"misleading", "tone", "platform", "industry_term", "consistency"}:
        score += 10
    if issue_type in {"chinglish", "grammar", "spelling"}:
        score += 6

    risky_words = {"delete", "discard", "subscribe", "payment", "purchase", "save", "export", "download", "failed", "permission", "access", "retry", "remove", "leave"}
    if any(w in normalize_text(current_text) for w in risky_words):
        score += 10

    score += min(repeated_count * 3, 15)

    if score >= 80:
        return "P0", score
    if score >= 60:
        return "P1", score
    if score >= 40:
        return "P2", score
    return "P3", score


def find_best_bbox(current_text: str, items: List[Dict[str, Any]]) -> List[int]:
    norm = normalize_text(current_text)
    best_bbox: List[int] = []
    best_score = 0.0
    for item in items:
        cand = normalize_text(item.get("text", ""))
        if not cand:
            continue
        if norm == cand:
            return item.get("bbox", []) or []
        score = fuzzy_similarity(norm, cand)
        if norm in cand or cand in norm:
            score += 0.08
        if score > best_score:
            best_score = score
            best_bbox = item.get("bbox", []) or []
    return best_bbox if best_score >= 0.72 else []


def search_code_locations(repo_root: Optional[str], current_text: str, max_hits: int = 5) -> List[Dict[str, Any]]:
    if not repo_root:
        return []
    root = Path(repo_root)
    if not root.exists():
        return []
    needle = current_text.strip()
    if not needle or len(needle) < 2:
        return []

    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_FILE_EXTS:
            continue
        try:
            if p.stat().st_size > 2_000_000:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if needle not in text:
            continue
        try:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if needle in line:
                    hits.append({
                        "file": str(p.relative_to(root)),
                        "line": lineno,
                        "snippet": line.strip()[:220],
                    })
                    if len(hits) >= max_hits:
                        return hits
        except Exception:
            continue
    return hits


def build_markdown_report(screen_reports: List[Dict[str, Any]], product_context_summary: str) -> str:
    lines = []
    if product_context_summary:
        lines.append("# 产品上下文概要")
        lines.append("")
        lines.append(product_context_summary)
        lines.append("")

    for screen in screen_reports:
        screen_name = screen.get("screen_name") or screen.get("image")
        purpose = screen.get("screen_purpose", "")
        lines.append(f"## {screen_name} — {purpose}".rstrip(" —"))
        lines.append("")
        lines.append(f"**用户旅程**：{screen.get('user_journey_from', '')} → **当前界面** → {screen.get('user_journey_to', '')}".strip())
        lines.append(f"**用户任务**：{screen.get('user_task', '')}")
        lines.append("")
        if screen.get("issues"):
            lines.append("### 发现的问题")
            lines.append("")
            lines.append("| # | 位置 | 当前文案 | 问题 | 建议修改 |")
            lines.append("|---|------|---------|------|---------|")
            for idx, issue in enumerate(screen["issues"], start=1):
                cur = str(issue.get("current_text", "")).replace("|", "\\|")
                prob = str(issue.get("problem", "")).replace("|", "\\|")
                sug = str(issue.get("suggestion", "")).replace("|", "\\|")
                pos = str(issue.get("position", "other")).replace("|", "\\|")
                lines.append(f"| {idx} | {pos} | \"{cur}\" | {prob} | \"{sug}\" |")
            lines.append("")
            lines.append("### 代码定位（如果有代码仓库）")
            if any(issue.get("code_locations") for issue in screen["issues"]):
                for issue in screen["issues"]:
                    for loc in issue.get("code_locations", [])[:3]:
                        lines.append(f"- `{issue.get('current_text', '')}` → 在 `{loc['file']}` 第 {loc['line']} 行找到")
            else:
                lines.append("- **未在 strings 文件中找到** — 可能是硬编码或后端下发")
        else:
            lines.append("此界面文案无问题")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def audit_folder(
    folder: str,
    output_dir: str,
    rule_path: Optional[str] = None,
    use_languagetool: bool = True,
    use_ollama: bool = True,
    lt_url: str = DEFAULT_LT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    skill_md: Optional[str] = None,
    product_context: str = "",
    app_name: str = "",
    platform: str = "mobile",
    repo_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rules = load_rules(rule_path)
    skill_text = load_skill_text(skill_md)
    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found in: {folder}")

    ocr_backend = OCRBackend()
    root = Path(folder)
    flat_results: List[Dict[str, Any]] = []
    screen_reports: List[Dict[str, Any]] = []
    all_visible_texts: List[str] = []

    # OCR all screens first for cross-screen frequency
    screen_payloads: List[Dict[str, Any]] = []
    for img in images:
        rel_path = str(img.relative_to(root))
        print(f"[OCR] {rel_path}")
        ocr_items = ocr_backend.predict(str(img))
        merged_items = merge_nearby_text_blocks(ocr_items)
        visible_items = []
        for item in merged_items:
            if not item.text.strip():
                continue
            visible_items.append({
                "text": item.text.strip(),
                "bbox": item.bbox,
                "ocr_confidence": item.confidence,
            })
            all_visible_texts.append(item.text.strip())
        screen_payloads.append({
            "image": rel_path,
            "image_name": img.name,
            "image_path": str(img.resolve()),
            "visible_items": visible_items,
        })

    freq = Counter(normalize_text(x) for x in all_visible_texts)
    overall_product_context = infer_product_context(app_name, product_context, platform, "", all_visible_texts[:20])
    product_context_summary = overall_product_context

    for screen in screen_payloads:
        visible_items = screen["visible_items"]
        screen_context = infer_product_context(app_name, product_context, platform, screen["image"], [x["text"] for x in visible_items])
        llm_review = skill_screen_review(
            image_name=screen["image_name"],
            image_rel_path=screen["image"],
            visible_items=visible_items,
            product_context=screen_context,
            platform=platform,
            skill_text=skill_text,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            enabled=use_ollama,
        )
        if not llm_review.get("issues") and not use_ollama:
            llm_review = fallback_screen_review(visible_items, rules, lt_url, use_languagetool)

        if llm_review.get("product_context_summary"):
            product_context_summary = llm_review.get("product_context_summary")

        screen_issues = []
        for issue in llm_review.get("issues", []):
            current_text = str(issue.get("current_text", "")).strip()
            bbox = find_best_bbox(current_text, visible_items)
            repeated_count = freq[normalize_text(current_text)] if current_text else 1
            priority, score = compute_priority_from_skill(issue, repeated_count, current_text)
            grammar_matches = check_with_languagetool(current_text, lt_url, enabled=use_languagetool) if current_text else []
            rule_hits = rule_match(current_text, rules) if current_text else []
            code_locations = search_code_locations(repo_root, current_text) if current_text else []

            row = {
                "image": screen["image"],
                "image_name": screen["image_name"],
                "image_path": screen["image_path"],
                "screen_name": llm_review.get("screen_name", screen["image_name"]),
                "screen_purpose": llm_review.get("screen_purpose", ""),
                "user_journey_from": llm_review.get("user_journey_from", ""),
                "user_journey_to": llm_review.get("user_journey_to", ""),
                "user_task": llm_review.get("user_task", ""),
                "product_context_summary": llm_review.get("product_context_summary", product_context_summary),
                "position": issue.get("position", "other"),
                "text": current_text,
                "bbox": bbox,
                "problem": issue.get("problem", ""),
                "suggestion": issue.get("suggestion", ""),
                "skill_severity": issue.get("severity", "minor"),
                "skill_issue_type": issue.get("issue_type", "other"),
                "skill_confidence": issue.get("confidence", 0.0),
                "priority": priority,
                "score": score,
                "repeated_count": repeated_count,
                "grammar_issue_count": len(grammar_matches),
                "grammar_matches": grammar_matches[:5],
                "rule_hits": rule_hits,
                "top_rule_category": rule_hits[0].get("category", "") if rule_hits else "",
                "top_rule_match_mode": rule_hits[0].get("match_mode", "") if rule_hits else "",
                "llm_verdict": "issues_found",
                "llm_reason": issue.get("problem", ""),
                "code_locations": code_locations,
            }
            flat_results.append(row)
            issue_copy = dict(issue)
            issue_copy["code_locations"] = code_locations
            screen_issues.append(issue_copy)

        screen_reports.append({
            "image": screen["image"],
            "image_path": screen["image_path"],
            "screen_name": llm_review.get("screen_name", screen["image_name"]),
            "screen_purpose": llm_review.get("screen_purpose", ""),
            "user_journey_from": llm_review.get("user_journey_from", ""),
            "user_journey_to": llm_review.get("user_journey_to", ""),
            "user_task": llm_review.get("user_task", ""),
            "product_context_summary": llm_review.get("product_context_summary", product_context_summary),
            "screen_verdict": llm_review.get("screen_verdict", "issues_found" if screen_issues else "no_issue"),
            "no_issue_summary": llm_review.get("no_issue_summary", ""),
            "issues": screen_issues,
            "visible_texts": [x["text"] for x in visible_items],
            "llm_error": llm_review.get("llm_error", ""),
        })

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(flat_results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(flat_results).to_csv(out_dir / "report.csv", index=False, encoding="utf-8-sig")
    (out_dir / "screen_report.json").write_text(json.dumps(screen_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    md = build_markdown_report(screen_reports, product_context_summary)
    (out_dir / "review_report.md").write_text(md, encoding="utf-8")
    print(f"Done. CSV: {(out_dir / 'report.csv')}")
    print(f"Done. JSON: {(out_dir / 'report.json')}")
    print(f"Done. Screen JSON: {(out_dir / 'screen_report.json')}")
    print(f"Done. Markdown: {(out_dir / 'review_report.md')}")
    return flat_results


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit screenshots using a skill-aligned UX copy review flow.")
    p.add_argument("--input", required=True, help="Folder containing screenshots")
    p.add_argument("--rules", default=None, help="Optional YAML rules file")
    p.add_argument("--output", default="output", help="Output folder")
    p.add_argument("--disable-languagetool", action="store_true", help="Disable local LanguageTool check")
    p.add_argument("--disable-ollama", action="store_true", help="Disable local Ollama review")
    p.add_argument("--lt-url", default=DEFAULT_LT_URL, help="LanguageTool HTTP endpoint")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama generate endpoint")
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    p.add_argument("--skill-md", default="", help="Optional path to the markdown skill file")
    p.add_argument("--product-context", default="", help="Manual product context to feed into review")
    p.add_argument("--app-name", default="", help="App name")
    p.add_argument("--platform", default="mobile", help="mobile / ios / android / web / desktop")
    p.add_argument("--repo-root", default="", help="Optional code repository root for string search")
    return p


def main():
    args = build_argparser().parse_args()
    audit_folder(
        folder=args.input,
        output_dir=args.output,
        rule_path=args.rules,
        use_languagetool=not args.disable_languagetool,
        use_ollama=not args.disable_ollama,
        lt_url=args.lt_url,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        skill_md=args.skill_md or None,
        product_context=args.product_context,
        app_name=args.app_name,
        platform=args.platform,
        repo_root=args.repo_root or None,
    )


if __name__ == "__main__":
    main()
