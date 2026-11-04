"""S6E11 official rules and metadata audit; READ-ONLY Kaggle API.

Never accepts terms, downloads files, trains, or submits.
A joined account is not by itself sufficient to lift training/submit gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

SLUG = "playground-series-s6e11"
OUTPUT = Path("state/s6e11_rules_review.json")
REPORT = Path("reports/s6e11_rules_review_latest.md")
ENTRY = Path("state/s6e11_entry_gate.json")


class StripHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fragments = []

    def handle_data(self, data):
        self.fragments.append(data)


def clean_html(s):
    p = StripHTML()
    p.feed(s or "")
    return re.sub(r"\s+", " ", " ".join(p.fragments)).strip()


def safe_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def parse_rule_text(body):
    """Only exact anchored expressions are facts; other fields stay unknown."""
    t = clean_html(body)
    def grab(pattern):
        match = re.search(pattern, t, re.I | re.S)
        return match.group(1).strip() if match else None
    limit = grab(r"(?:maximum|up to)\s+(\w+\s*\(\d+\)|\d+)\s+Submissions\s+per\s+day")
    group = grab(r"maximum Team size is\s+(\w+\s*\(\d+\)|\d+)")
    final = grab(r"select up to\s+(\w+\s*\(\d+\)|\d+)\s+Final Submissions")
    sections = {}
    for key, pat in (
        ("external_data", r"EXTERNAL DATA(?: AND TOOLS)?"),
        ("team_limits", r"TEAM LIMITS"),
        ("submission_limits", r"SUBMISSION LIMITS"),
        ("data_license", r"DATA ACCESS AND USE"),
        ("eligibility", r"ELIGIBILITY"),
    ):
        m = re.search(pat, t, re.I)
        sections[key] = t[m.start():m.start()+1150] if m else None
    return {
        "content_sha256": hashlib.sha256((body or "").encode("utf-8")).hexdigest() if body else None,
        "content_length": len(body or ""),
        "rules_full_text_retrieved": bool(len(t) > 4000 and sections["external_data"] and sections["submission_limits"]),
        "submission_limit_expression": limit,
        "team_limit_expression": group,
        "final_selection_expression": final,
        "section_excerpts": sections,
    }


def query_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    result = {
        "pages": {"status": "not_checked"},
        "data_files": {"status": "not_checked"},
        "submission_limits": {"status": "not_checked"},
        "account_submissions": {"status": "not_checked"},
    }
    try:
        pages = api.competition_list_pages(SLUG) or []
        names = sorted({str(getattr(p, "name", "") or "") for p in pages})
        rules = [p for p in pages if str(getattr(p, "name", "")).lower() == "rules"]
        parsed = parse_rule_text(getattr(rules[0], "content", "") or "") if len(rules)==1 else None
        result["pages"] = {"status": "success", "page_names": names, "rules": parsed}
    except Exception as exc:
        result["pages"] = {"status": "failed", "error_class": type(exc).__name__}
    try:
        files = api.competition_list_files(SLUG, page_size=100)
        entries = []
        for f in getattr(files, "files", []) or []:
            entries.append({"name": str(getattr(f,"name","")),
                            "total_bytes": safe_int(getattr(f,"total_bytes",None))})
        result["data_files"] = {"status":"success", "files":entries,
                                "paging_more":bool(getattr(files, "next_page_token", None))}
    except Exception as exc:
        result["data_files"] = {"status":"failed", "error_class":type(exc).__name__}
    try:
        limits = api.competition_get_submission_limits(SLUG)
        result["submission_limits"] = {
            "status": "success", "num_today":safe_int(getattr(limits,"num_today",None)),
            "num_total":safe_int(getattr(limits,"num_total",None)),
            "num_allowed_now":safe_int(getattr(limits,"num_allowed_now",None)),
            "limited_by_total":bool(getattr(limits,"limited_by_total",False)),
        }
    except Exception as exc:
        result["submission_limits"] = {"status":"failed", "error_class":type(exc).__name__}
    try:
        subs = api.competition_submissions(SLUG,page_size=20) or []
        result["account_submissions"] = {"status":"success","visible_count_first_page":len(subs),
                                         "has_any":bool(subs)}
    except Exception as exc:
        result["account_submissions"] = {"status":"failed", "error_class":type(exc).__name__}
    return result


def format_report(o):
    p=o.get("pages",{})
    rules=p.get("rules") or {}
    files=o.get("data_files",{})
    lim=o.get("submission_limits",{})
    auth=o.get("account_submissions",{})
    status=o.get("status")
    def show(x):
        return str(x) if x is not None and x != "" else "待核实"
    lines=[
        "# Kaggle S6E11 官方规则只读复核",
        "",
        f"- Kaggle 参赛认证：{show(o.get('joined'))}，来自已有认证门禁记录（此项未重新接受规则）",
        f"- 复核状态：**{status}**",
        "- 原则：没有官方 S6E11 专属规则正文的完整证据，不自动批准训练、数据下载、外部模型或提交",
        "- 禁止动作：自动接受规则、训练、下载受限数据、正式提交、使用付费资源",
        "",
        "## 官网已公开确认",
        "- 比赛：Predicting Credit Card Fraud（Playground Series S6E11）",
        "- 指标：ROC AUC；正式提交列：\`id,is_fraud\`，预测概率在 [0,1]",
        "- 最终期限：2026-11-30 23:59 UTC（以最新官方赛程为准）",
        "- 奖励：前三名周边；不授 Kaggle Points / Medals；2026 同系列每人仅有一次周边奖励",
        "- 公开资料：https://www.kaggle.com/competitions/playground-series-s6e11/overview",
        "",
        "## Kaggle 认证元数据查询",
        f"- 官方 Page API：{p.get('status','unknown')}，页面名：{p.get('page_names',[])}",
        f"- 是否取得完整专属 rules 内容：{rules.get('rules_full_text_retrieved',False)}",
        f"- 比赛规则 SHA256：{show(rules.get('content_sha256'))}",
        f"- 专属队伍上限表达：{show(rules.get('team_limit_expression'))}",
        f"- 专属每日提交限制表达：{show(rules.get('submission_limit_expression'))}",
        f"- 专属最终计分选中数量表达：{show(rules.get('final_selection_expression'))}",
        f"- 数据清单查询：{files.get('status','unknown')}；仅文件元信息，没有下载",
        f"- 文件元信息：{files.get('files',[])}",
        f"- 文件列表是否可能有下一页：{show(files.get('paging_more'))}",
        f"- 官方账户当下提交额度接口：{lim}",
        f"- 已有正式提交历史接口：{auth}",
        "",
        "## 专属规则可核查片段（非完整法律文本）",
    ]
    for k,v in (rules.get("section_excerpts") or {}).items():
        lines += [f"### {k}", "", v or "未取得：待核实", ""]
    lines += [
        "## 当前门禁与待补证",
        "- 账户加入不代表已经逐项审计所有比赛特定条款；本报告不替用户接受协议。",
        "- 专属规则全文、资格及地域限制、外部数据、预训练模型、生成式 AI 使用许可：未明确验证前不擅自允许。",
        "- 数据集许可证、实际文件格式/哈希、行数及目标分布：需要先通过获准的官方只读元数据/许可核验；目前不可假定。",
        "- 训练只使用比赛允许的官方数据和零增费资源；任何不明确环节都 fail-closed。",
        "- 后续：验证完成后另行办理独立数据/OOF/提交门禁；本次不启动训练，也不修改原报名门禁。",
        "",
        f"- 查询阶段：{show(o.get('failed_stage'))}",
        f"- 异常类（不包含凭据）：{show(o.get('error_class'))}",
        "",
    ]
    return "\n".join(lines)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test",action="store_true")
    opts=ap.parse_args()
    if opts.test:
        r=parse_rule_text("x"*4100+" EXTERNAL DATA AND TOOLS SUBMISSION LIMITS maximum Team size is three (3).")
        assert r["team_limit_expression"] == "three (3)"
        assert parse_rule_text("")["rules_full_text_retrieved"] is False
        print("s6e11 rules review offline checks passed")
        return
    entry=json.loads(ENTRY.read_text(encoding="utf8")) if ENTRY.exists() else {}
    o={"competition":SLUG,"joined":entry.get("userHasEntered",False),
       "deadline":entry.get("deadline"),
       "status":"review_pending_specific_rules",
       "training_allowed":False,"submission_allowed":False,
       "rules_review_passed":False}
    if not os.environ.get("KAGGLE_API_TOKEN"):
        o.update(status="blocked_missing_credential",failed_stage="preflight")
    elif entry.get("userHasEntered") is not True:
        o.update(status="blocked_no_authenticated_join",failed_stage="entry_gate")
    else:
        try:
            o.update(query_api())
            rules=(o.get("pages",{}).get("rules") or {})
            if rules.get("rules_full_text_retrieved"):
                o["status"]="rules_text_fetched_manual_clause_audit_required"
            else:
                o["status"]="review_pending_specific_rules"
        except Exception as exc:
            o.update(status="review_query_failed",failed_stage="kaggle_api",
                     error_class=type(exc).__name__)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    REPORT.write_text(format_report(o),encoding="utf8")
    print("S6E11 review:",o["status"])
    print("S6E11 pages query:",o.get("pages",{}).get("status"))
    print("S6E11 data file metadata query:",o.get("data_files",{}).get("status"))
    print("S6E11 submission limits query:",o.get("submission_limits",{}).get("status"))
    print("S6E11 safe gate: training=False, submission=False")


if __name__=="__main__":
    main()
