"""Read public Kaggle leaderboard top three through authenticated official CLI."""
import csv
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SLUG="playground-series-s6e10"

def run(args):
    proc=subprocess.run(args,text=True,capture_output=True,timeout=120)
    if proc.returncode:
        raise RuntimeError(f"Official CLI command failed ({args[-2:]}): {(proc.stderr or proc.stdout)[-700:]}")
    return proc.stdout

def parse(text):
    """Return only explicit rank 1/2/3 rows, never infer rank from score order."""
    value=text.strip()
    if value.startswith('Next Page Token ='):
        value='\n'.join(value.splitlines()[1:]).strip()
    if not value:
        raise ValueError("Empty Kaggle leaderboard")
    rows=[]
    try:
        candidate=json.loads(value)
        if isinstance(candidate,list):rows=candidate
        elif isinstance(candidate,dict):
            for key in ("results","leaderboard","items","entries"):
                if isinstance(candidate.get(key),list):
                    rows=candidate[key];break
    except json.JSONDecodeError:
        pass
    if not rows:
        content=list(csv.DictReader(io.StringIO(value)))
        if content and len(content)>2:
            rows=content
    if not rows:
        return []
    # The authenticated Kaggle --show leaderboard is already ordered by rank.
    # Derive ordinal position from OFFICIAL returned order, never sort by score.
    if len(rows)>=3 and all(isinstance(row,dict) and all(k in row for k in ('teamId','teamName','score')) for row in rows[:3]):
        if len({str(row['teamId']) for row in rows[:3]})==3:
            return [{'rank':i+1,'team':str(row['teamName']).strip(),
                     'team_id':row['teamId'],'public_score':str(row['score']),
                     'rank_basis':'official_api_row_order'} for i,row in enumerate(rows[:3])]
    found={}
    for row in rows:
        if not isinstance(row,dict):continue
        raw=next((row[key] for key in ("rank","Rank","publicRank","public_rank") if row.get(key) is not None),None)
        try:rank=int(str(raw).strip())
        except (TypeError,ValueError):continue
        if rank not in (1,2,3):continue
        team=next((str(row.get(key)).strip() for key in ("teamName","team","TeamName","Team","name","team_name") if row.get(key)),None)
        score=next((str(row.get(key)).strip() for key in ("score","Score","publicScore","public_score") if row.get(key) is not None),None)
        if not team or not score:continue
        found[rank]={"rank":rank,"team":team,"public_score":score}
    if set(found)!={1,2,3}:return []
    return [found[i] for i in (1,2,3)]

def main():
    checked=datetime.now(timezone.utc).isoformat()
    attempts=[]
    leaderboard=[]
    for args in (
        ["kaggle","competitions","leaderboard",SLUG,"--show","--format","json"],
        ["kaggle","competitions","leaderboard",SLUG,"--show","--csv"],
    ):
        try:
            raw=run(args)
            leaderboard=parse(raw)
            attempts.append({"mode":args[-1],"parsed":bool(leaderboard),"public_output_preview":raw[:900] if not leaderboard else ""})
            if leaderboard:break
        except (RuntimeError,ValueError) as exc:
            attempts.append({"mode":args[-1],"error":str(exc)[:600]})
    verified=bool(leaderboard)
    status={"checked_at":checked,"competition":SLUG,"official_source":"kaggle_authenticated_competitions_leaderboard",
            "top3_verified":verified,"top3":leaderboard,"method_verified":False,
            "method_note":"Official public rank proves team/score, not private code or technique.",
            "attempts":attempts,"leaderboard_url":f"https://www.kaggle.com/competitions/{SLUG}/leaderboard"}
    (ROOT/"state").mkdir(exist_ok=True)
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"state/v2_9_4_top3.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# S6E10｜官方 Public 排行榜前三核查","",f"- UTC：{checked}",
           f"- 官方账户认证：{'成功' if verified else '未核实（已保留尝试日志）'}",
           "- 注意：当前 Public 前三不是最终 Private 领奖名单。","",
           "| Public 名次 | Team | Public AUC |","|---:|---|---:|"]
    for r in leaderboard:lines.append(f"| {r['rank']} | {r['team'].replace('|','/')} | {r['public_score']} |")
    if not leaderboard:lines.append("| 未核实 | — | — |")
    lines.extend(["",f"- 官方榜单：{status['leaderboard_url']}","",
                  "## 方法证据","",
                  "排行榜仅证明队伍与分数。前三名的具体方法必须由该队伍公开的 Kaggle 讨论、Notebook 或可验证代码支持；不能把第三方复盘冒充榜一方案。",
                  "","## CLI 诊断","",json.dumps(attempts,ensure_ascii=False,indent=2)])
    (ROOT/"reports/v2_9_4_top3_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if not verified:print("WARNING: top3 not yet verified; no identities inferred.")

if __name__=="__main__":
    main()
