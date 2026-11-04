"""Generic integrity-only CSV audit for an already-joined Kaggle competition.

No fitting, model code, external data, label use for prediction, or submissions.
Raw competition data is kept in a disposable GitHub runner directory and NEVER
uploaded to GitHub artifacts or committed.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.s6e11_entry_gate import exact, rows_for
from kaggle_agent.scout import parse_deadline

SLUG = "playground-series-s6e11"
ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "state/s6e11_rule_approval.json"
ENTRY = ROOT / "state/s6e11_entry_gate.json"
RULE_STATE = ROOT / "state/s6e11_rules_review.json"
RESULT = ROOT / "state/s6e11_data_integrity.json"
REPORT = ROOT / "reports/s6e11_data_integrity_latest.md"
EXPECTED = frozenset(("train.csv", "test.csv", "sample_submission.csv"))
STAGE = "init"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def preflight():
    global STAGE
    STAGE = "local_rules"
    approved = read_json(RULES)
    membership = read_json(ENTRY)
    prior = read_json(RULE_STATE)
    assert approved["competition"] == SLUG and approved["allow_download_competition_data_for_audit"] is True
    assert approved["review_scope"] == "official_competition_data_only_no_external_data_no_unreviewed_models"
    assert approved["allow_training_without_separate_data_gate"] is False
    assert approved["allow_submission_without_oof_and_quota_gate"] is False
    assert prior["rules_review_passed"] is True and prior["training_allowed"] is False
    assert prior["submission_allowed"] is False and prior["joined"] is True
    assert prior["pages"]["rules"]["content_sha256"] == approved["source_rules_sha256"]
    assert membership["userHasEntered"] is True
    assert prior["competition"] == SLUG
    assert os.environ.get("KAGGLE_API_TOKEN"), "credential absent"
    STAGE = "fresh_kaggle_membership"
    general, _ = rows_for("general", "playground", 3)
    entered, complete = rows_for("entered", "all", 5)
    g, e = exact(general), exact(entered)
    assert complete and g is not None and e is not None
    assert g.get("userHasEntered") is True and e.get("userHasEntered") is True
    deadline = parse_deadline(g.get("deadline"))
    assert deadline is not None and deadline > datetime.now(timezone.utc)
    assert deadline.isoformat() == membership["deadline"]
    STAGE = "fresh_kaggle_rules_and_file_list"
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    pages = api.competition_list_pages(SLUG) or []
    rules = [p for p in pages if str(getattr(p, "name", "")).lower() == "rules"]
    assert len(rules) == 1 and getattr(rules[0], "content", None)
    digest = hashlib.sha256(rules[0].content.encode("utf-8")).hexdigest()
    assert digest == approved["source_rules_sha256"]
    files = api.competition_list_files(SLUG, page_size=100)
    assert not getattr(files, "next_page_token", None)
    manifest = {str(f.name): int(f.total_bytes) for f in files.files}
    assert frozenset(manifest) == EXPECTED
    prior_files = {r["name"]: r["total_bytes"] for r in prior["data_files"]["files"]}
    assert manifest == prior_files
    return {"rules_sha256": digest, "deadline_utc": deadline.isoformat(),
            "source_file_sizes": manifest}


def file_sha_and_rows(z, filename, required, expected_columns=None):
    global STAGE
    STAGE = "scan_" + filename.replace(".", "_")
    sha = hashlib.sha256()
    # ZipExtFile is non-seekable. Stream it once; never extract or upload raw CSV.
    stream = z.open(filename, "r")
    class HashedStream(io.RawIOBase):
        def readable(self): return True
        def readinto(self, dest):
            block = stream.read(len(dest))
            n = len(block)
            if n:
                dest[:n] = block
                sha.update(block)
            return n
    text_stream = io.TextIOWrapper(io.BufferedReader(HashedStream(), buffer_size=1024*1024),
                                   encoding="utf-8-sig", newline="")
    reader = csv.DictReader(text_stream)
    fields = reader.fieldnames or []
    assert fields and len(fields) == len(set(fields))
    assert all(c and c.strip() == c for c in fields)
    assert all(c in fields for c in required)
    if expected_columns is not None:
        assert fields == expected_columns, "sample submission schema mismatch"
    missing = Counter()
    ids = set()
    target = Counter()
    count = 0
    duplicate_ids = 0
    malformed = 0
    sample_ids_in_order = []
    for row in reader:
        count += 1
        if None in row or any(v is None for v in row.values()):
            malformed += 1
            continue
        idval = row.get("id")
        if idval in (None, ""):
            malformed += 1
        elif idval in ids:
            duplicate_ids += 1
        else:
            ids.add(idval)
            if filename == "sample_submission.csv":
                sample_ids_in_order.append(idval)
        for c, value in row.items():
            if value is None or not value.strip():
                missing[c] += 1
        if filename == "train.csv":
            y = row.get("is_fraud", "")
            if y in ("0", "0.0"): target["0"] += 1
            elif y in ("1", "1.0"): target["1"] += 1
            else: target["invalid"] += 1
    text_stream.close()
    return {
        "file": filename, "sha256": sha.hexdigest(), "row_count": count,
        "columns": fields, "missing_by_column": dict(missing),
        "duplicate_ids": duplicate_ids, "malformed_rows": malformed,
        "target_counts": dict(target) if filename == "train.csv" else None,
    }, ids, sample_ids_in_order


def audit_archive(zip_path, manifest):
    global STAGE
    STAGE = "open_official_archive"
    with zipfile.ZipFile(zip_path) as z:
        entries = z.infolist()
        assert len(entries) == 3
        assert frozenset(x.filename for x in entries) == EXPECTED
        assert all(not x.is_dir() and x.file_size == manifest[x.filename] for x in entries)
        assert all(x.file_size < 500_000_000 for x in entries)
        train, train_ids, _ = file_sha_and_rows(z, "train.csv", ("id", "is_fraud"))
        test, test_ids, _ = file_sha_and_rows(z, "test.csv", ("id",))
        sample, sample_ids, _ = file_sha_and_rows(z, "sample_submission.csv",
                                                  ("id", "is_fraud"),
                                                  expected_columns=["id", "is_fraud"])
    STAGE = "cross_file_checks"
    assert len(train["columns"]) >= 3
    assert [x for x in train["columns"] if x != "is_fraud"] == test["columns"]
    assert "is_fraud" not in test["columns"]
    assert train_ids.isdisjoint(test_ids)
    assert len(sample_ids) == len(test_ids)
    assert set(sample_ids) == test_ids
    assert all(x["row_count"] > 0 and x["duplicate_ids"] == 0
               and x["malformed_rows"] == 0 for x in (train, test, sample))
    assert train["target_counts"].get("invalid", 0) == 0
    assert train["target_counts"].get("0", 0) > 0 and train["target_counts"].get("1", 0) > 0
    assert sum(train["target_counts"].values()) == train["row_count"]
    return {"train": train, "test": test, "sample_submission": sample,
            "train_test_id_overlap": 0, "submission_id_set_exact": True,
            "target_positive_rate": train["target_counts"]["1"] / train["row_count"],
            "data_integrity_passed": True}


def persist(result):
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+"\n",encoding="utf-8")
    lines = [
        "# S6E11 真实数据完整性只读验收",
        "", "- 竞赛：" + SLUG, "- 状态：" + str(result["status"]),
        "- 当前验收阶段：" + str(result.get("failed_stage") or "finished"),
        "- Kaggle 规则哈希：" + str(result.get("rules_sha256") or "unknown"),
        "- 额外付费：禁止（GitHub 公共仓库已有 Actions 环境，未使用云端付费算力）",
        "- 本流程不训练、不下载外部数据、不提交、不写入训练代码/样本/预测到公共仓库",
    ]
    if result.get("data_integrity_passed"):
        for name in ("train","test","sample_submission"):
            f = result[name]
            lines += ["", "## " + f["file"],
                      "- SHA256：" + f["sha256"],
                      "- 行数：" + str(f["row_count"]),
                      "- 列：" + json.dumps(f["columns"],ensure_ascii=False),
                      "- 缺失统计：" + json.dumps(f["missing_by_column"],ensure_ascii=False)]
        lines += ["", "- 欺诈样本占比：" + str(result["target_positive_rate"]),
                  "- train/test ID 无交叉，sample ID 集合与 test 精确相等：是"]
    else:
        lines += ["", "- 安全失败，仅记录异常类型（不记录敏感异常文本）：" +
                  str(result.get("error_class") or "unverified")]
    lines += ["", "此结果仅证明数据结构/文件完整性，不能证明无目标泄漏、分布无偏或模型可提交。",
              "后续训练/提交须独立的 CV/OOF 和来源 SHA 门禁。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("S6E11 audit stage:",result["status"])
    print("S6E11 data integrity passed:",bool(result.get("data_integrity_passed")))


def main():
    global STAGE
    result={"competition":SLUG,"status":"blocked_data_audit_pending",
            "training_allowed":False,"submission_allowed":False,
            "data_integrity_passed":False}
    try:
        result.update(preflight())
        STAGE = "download_official_competition_only"
        with tempfile.TemporaryDirectory(prefix="kaggle_s6e11_") as directory:
            cmd=["kaggle","competitions","download","-c",SLUG,"-p",directory]
            proc=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=240,
                                check=False)
            assert proc.returncode == 0, "official download failed"
            paths = list(Path(directory).glob("*.zip"))
            assert len(paths) == 1 and paths[0].stat().st_size > 0
            result.update(audit_archive(paths[0],result["source_file_sizes"]))
            result["status"]="data_integrity_verified"
    except Exception as exc:
        result.update(status="blocked_data_audit_failed",data_integrity_passed=False,
                      failed_stage=STAGE,error_class=type(exc).__name__)
    persist(result)


if __name__=="__main__":
    main()
