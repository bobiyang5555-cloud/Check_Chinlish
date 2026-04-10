import json
from pathlib import Path
from typing import Any, List

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="UI Copy Auditor", layout="wide")


def load_data(report_json: Path, report_csv: Path) -> pd.DataFrame:
    if report_json.exists():
        data = json.loads(report_json.read_text(encoding="utf-8"))
        return pd.DataFrame(data)
    if report_csv.exists():
        return pd.read_csv(report_csv)
    return pd.DataFrame()


def safe_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    return []


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


def get_first_rule_field(rule_hits: Any, field: str) -> str:
    hits = safe_list(rule_hits)
    if hits and isinstance(hits[0], dict):
        return str(hits[0].get(field, "") or "")
    return ""


def draw_bbox(img: Image.Image, bbox):
    if isinstance(bbox, list) and len(bbox) == 4:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline="red", width=4)
        return img
    return img


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "top_rule_category" not in df.columns:
        df["top_rule_category"] = df.get("rule_hits", pd.Series([[]] * len(df))).apply(
            lambda x: get_first_rule_field(x, "category")
        )

    if "top_rule_match_mode" not in df.columns:
        df["top_rule_match_mode"] = df.get("rule_hits", pd.Series([[]] * len(df))).apply(
            lambda x: get_first_rule_field(x, "match_mode")
        )

    if "bbox" in df.columns:
        df["bbox"] = df["bbox"].apply(parse_bbox)

    if "priority" not in df.columns:
        df["priority"] = "P3"
    if "score" not in df.columns:
        df["score"] = 0
    if "suggestion" not in df.columns:
        df["suggestion"] = ""
    if "text" not in df.columns:
        df["text"] = ""
    if "image" not in df.columns:
        df["image"] = ""
    if "image_path" not in df.columns:
        df["image_path"] = ""
    if "llm_verdict" not in df.columns:
        df["llm_verdict"] = ""
    if "llm_reason" not in df.columns:
        df["llm_reason"] = ""
    if "repeated_count" not in df.columns:
        df["repeated_count"] = 1
    if "grammar_issue_count" not in df.columns:
        df["grammar_issue_count"] = 0

    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    df["priority_order"] = df["priority"].map(priority_order).fillna(99)
    return df


def main():
    st.title("UI Copy Auditor")
    st.caption("本地截图英文质检器")

    default_report_json = Path("output/report.json")
    default_report_csv = Path("output/report.csv")

    with st.sidebar:
        st.header("数据源")
        report_json_input = st.text_input("report.json 路径", value=str(default_report_json))
        report_csv_input = st.text_input("report.csv 路径", value=str(default_report_csv))
        st.markdown("先运行主程序生成报告，再打开这个页面。")

    report_json = Path(report_json_input)
    report_csv = Path(report_csv_input)
    df = normalize_dataframe(load_data(report_json, report_csv))

    if df.empty:
        st.warning("没有找到 report.json 或 report.csv。先运行主程序生成输出。")
        st.code("python ui_copy_auditor.py --input screenshots --rules term_rules.yaml --output output")
        st.stop()

    with st.sidebar:
        st.header("筛选")
        priority_filter = st.multiselect(
            "优先级",
            options=["P0", "P1", "P2", "P3"],
            default=["P0", "P1", "P2", "P3"],
        )

        category_options = sorted([x for x in df["top_rule_category"].fillna("").unique().tolist() if x])
        category_filter = st.multiselect(
            "分类",
            options=category_options,
            default=category_options,
        )

        match_mode_options = sorted([x for x in df["top_rule_match_mode"].fillna("").unique().tolist() if x])
        match_mode_filter = st.multiselect(
            "命中方式",
            options=match_mode_options,
            default=match_mode_options,
        )

        verdict_options = sorted([x for x in df["llm_verdict"].fillna("").unique().tolist() if x])
        verdict_filter = st.multiselect(
            "LLM判断",
            options=verdict_options,
            default=verdict_options,
        )

        keyword = st.text_input("关键词搜索")
        only_with_suggestion = st.checkbox("只看有替代建议", value=False)
        only_high_confidence = st.checkbox("只看高分项（score >= 60）", value=False)

    filtered = df[df["priority"].isin(priority_filter)].copy()

    if category_options:
        filtered = filtered[filtered["top_rule_category"].fillna("").isin(category_filter)]
    if match_mode_options:
        filtered = filtered[filtered["top_rule_match_mode"].fillna("").isin(match_mode_filter)]
    if verdict_options:
        filtered = filtered[filtered["llm_verdict"].fillna("").isin(verdict_filter)]
    if keyword:
        filtered = filtered[
            filtered["text"].astype(str).str.contains(keyword, case=False, na=False)
            | filtered["suggestion"].astype(str).str.contains(keyword, case=False, na=False)
            | filtered["image"].astype(str).str.contains(keyword, case=False, na=False)
        ]
    if only_with_suggestion:
        filtered = filtered[filtered["suggestion"].fillna("") != ""]
    if only_high_confidence:
        filtered = filtered[pd.to_numeric(filtered["score"], errors="coerce").fillna(0) >= 60]

    filtered = filtered.sort_values(["priority_order", "score"], ascending=[True, False])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总条数", len(df))
    c2.metric("筛选后", len(filtered))
    c3.metric("P0", int((df["priority"] == "P0").sum()))
    c4.metric("P1", int((df["priority"] == "P1").sum()))

    st.subheader("结果总览")
    show_cols = [
        col for col in [
            "priority", "score", "top_rule_category", "top_rule_match_mode",
            "llm_verdict", "text", "suggestion", "image"
        ] if col in filtered.columns
    ]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

    if report_csv.exists():
        st.download_button(
            "下载 CSV",
            data=report_csv.read_bytes(),
            file_name=report_csv.name,
            mime="text/csv",
        )
    if report_json.exists():
        st.download_button(
            "下载 JSON",
            data=report_json.read_bytes(),
            file_name=report_json.name,
            mime="application/json",
        )

    st.subheader("逐条查看")
    for _, row in filtered.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 1.25])

            with col1:
                img_path = Path(str(row.get("image_path", "")))
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    img = draw_bbox(img, row.get("bbox", []))
                    st.image(img, caption=str(row.get("image", "")), use_container_width=True)
                else:
                    st.info(f"找不到图片: {img_path}")

            with col2:
                st.markdown(f"### {row.get('text', '')}")
                st.write(
                    f"**优先级：** {row.get('priority', '-') }  |  "
                    f"**分数：** {row.get('score', '-') }  |  "
                    f"**分类：** {row.get('top_rule_category', '-') or '-'}  |  "
                    f"**命中方式：** {row.get('top_rule_match_mode', '-') or '-'}"
                )
                st.write(f"**建议改法：** {row.get('suggestion', '') or '-'}")
                st.write(f"**LLM判断：** {row.get('llm_verdict', '') or '-'}")
                st.write(f"**原因：** {row.get('llm_reason', '') or '-'}")
                st.write(f"**重复出现次数：** {row.get('repeated_count', 1)}")
                st.write(f"**Grammar问题数：** {row.get('grammar_issue_count', 0)}")
                rule_hits = row.get("rule_hits", [])
                if isinstance(rule_hits, list) and rule_hits:
                    st.write("**规则命中：**")
                    st.json(rule_hits)

            st.divider()


if __name__ == "__main__":
    main()
