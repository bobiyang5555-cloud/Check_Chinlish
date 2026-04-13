import json
from pathlib import Path
from typing import Any, List

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="UI Copy Reviewer (Skill)", layout="wide")


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def parse_bbox(v: Any):
    if isinstance(v, list) and len(v) == 4:
        return v
    if isinstance(v, str):
        try:
            obj = json.loads(v)
            if isinstance(obj, list) and len(obj) == 4:
                return obj
        except Exception:
            return []
    return []


def draw_bbox(img: Image.Image, bbox):
    if isinstance(bbox, list) and len(bbox) == 4:
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline="red", width=4)
    return img


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "bbox" in df.columns:
        df["bbox"] = df["bbox"].apply(parse_bbox)
    if "priority" not in df.columns:
        df["priority"] = "P3"
    if "score" not in df.columns:
        df["score"] = 0
    if "skill_severity" not in df.columns:
        df["skill_severity"] = "minor"
    if "skill_issue_type" not in df.columns:
        df["skill_issue_type"] = "other"
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    df["priority_order"] = df["priority"].map(order).fillna(99)
    return df


def main():
    st.title("UI Copy Reviewer (Skill-aligned)")
    st.caption("按 screen 上下文输出的 UI 文案审校结果")

    with st.sidebar:
        out_dir = st.text_input("输出目录", value="output")
    out_path = Path(out_dir)
    report_json = out_path / "report.json"
    screen_json = out_path / "screen_report.json"
    md_path = out_path / "review_report.md"

    rows = load_json(report_json, [])
    screens = load_json(screen_json, [])
    df = normalize_df(pd.DataFrame(rows))

    if df.empty and not screens:
        st.warning("没找到 skill 审校结果。先运行 ui_copy_auditor.py。")
        st.code("python ui_copy_auditor.py --input screenshots --output output --disable-ollama")
        st.stop()

    with st.sidebar:
        pri = st.multiselect("优先级", ["P0", "P1", "P2", "P3"], default=["P0", "P1", "P2", "P3"])
        sev_opts = sorted([x for x in df.get("skill_severity", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x]) if not df.empty else []
        sev = st.multiselect("严重程度", sev_opts, default=sev_opts)
        issue_opts = sorted([x for x in df.get("skill_issue_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x]) if not df.empty else []
        issue_types = st.multiselect("问题类型", issue_opts, default=issue_opts)
        keyword = st.text_input("关键词")
        only_suggestion = st.checkbox("只看有建议改法", value=False)

    if not df.empty:
        filtered = df[df["priority"].isin(pri)].copy()
        if sev_opts:
            filtered = filtered[filtered["skill_severity"].isin(sev)]
        if issue_opts:
            filtered = filtered[filtered["skill_issue_type"].isin(issue_types)]
        if keyword:
            filtered = filtered[
                filtered["text"].astype(str).str.contains(keyword, case=False, na=False)
                | filtered["suggestion"].astype(str).str.contains(keyword, case=False, na=False)
                | filtered["image"].astype(str).str.contains(keyword, case=False, na=False)
            ]
        if only_suggestion:
            filtered = filtered[filtered["suggestion"].fillna("") != ""]
        filtered = filtered.sort_values(["priority_order", "score"], ascending=[True, False])
    else:
        filtered = pd.DataFrame()

    if screens:
        ctx = screens[0].get("product_context_summary", "")
        if ctx:
            st.subheader("产品上下文")
            st.info(ctx)

    c1, c2, c3 = st.columns(3)
    c1.metric("Screens", len(screens))
    c2.metric("Issues", len(df))
    c3.metric("Filtered", len(filtered))

    if md_path.exists():
        st.download_button("下载 Markdown 审校报告", data=md_path.read_bytes(), file_name=md_path.name, mime="text/markdown")

    if not filtered.empty:
        st.subheader("问题总览")
        st.dataframe(filtered[[c for c in ["priority", "skill_severity", "skill_issue_type", "text", "suggestion", "image"] if c in filtered.columns]], use_container_width=True, hide_index=True)

    st.subheader("按界面查看")
    for screen in screens:
        with st.expander(f"{screen.get('screen_name', screen.get('image', 'screen'))} · {screen.get('screen_purpose', '')}", expanded=False):
            st.markdown(f"**用户旅程**：{screen.get('user_journey_from', '')} → **当前界面** → {screen.get('user_journey_to', '')}")
            st.markdown(f"**用户任务**：{screen.get('user_task', '')}")
            if screen.get("screen_verdict") == "no_issue":
                st.success(screen.get("no_issue_summary", "此界面文案无问题"))
            img_path = Path(screen.get("image_path", ""))
            if img_path.exists():
                st.image(Image.open(img_path).convert("RGB"), caption=screen.get("image", ""), use_container_width=True)

            issues = screen.get("issues", []) or []
            if issues:
                issue_df = pd.DataFrame(issues)
                if not issue_df.empty:
                    st.dataframe(issue_df[[c for c in ["position", "current_text", "problem", "suggestion", "severity", "issue_type"] if c in issue_df.columns]], use_container_width=True, hide_index=True)
                for issue in issues:
                    cur = issue.get("current_text", "")
                    st.markdown(f"- **{cur}** → `{issue.get('suggestion', '')}` · {issue.get('problem', '')}")
                    if issue.get("code_locations"):
                        for loc in issue["code_locations"][:3]:
                            st.caption(f"{loc['file']}:{loc['line']} · {loc['snippet']}")

    if not filtered.empty:
        st.subheader("逐条问题")
        for _, row in filtered.iterrows():
            with st.container():
                left, right = st.columns([1, 1.3])
                with left:
                    img_path = Path(str(row.get("image_path", "")))
                    if img_path.exists():
                        img = Image.open(img_path).convert("RGB")
                        img = draw_bbox(img, row.get("bbox", []))
                        st.image(img, caption=row.get("image", ""), use_container_width=True)
                with right:
                    st.markdown(f"### {row.get('text', '')}")
                    st.write(f"**建议修改：** {row.get('suggestion', '') or '-'}")
                    st.write(f"**问题：** {row.get('problem', '') or '-'}")
                    st.write(f"**优先级：** {row.get('priority', '-') } | **严重程度：** {row.get('skill_severity', '-') } | **问题类型：** {row.get('skill_issue_type', '-') }")
                    if isinstance(row.get('code_locations', None), list) and row.get('code_locations'):
                        st.write("**代码定位：**")
                        st.json(row.get('code_locations'))
                st.divider()


if __name__ == "__main__":
    main()
