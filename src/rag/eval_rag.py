"""Run and score R0 / R1 / R2.

usage:  python eval_rag.py dev    # R1 with k in {3, 5, 10} on the 10 dev questions -> choose k
        python eval_rag.py test   # R0, R1, R2 once on the 20 locked test questions with the chosen k

Automatic scoring is a transparent rule set (below); every test answer is then read by hand
(outputs/tables/rag_manual_review.csv) and the MANUAL label is the final one. No LLM judge.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, OUT_TABLES, PROTOCOL, sha256, save_json  # noqa: E402
from kb import load_records  # noqa: E402

RUNS = OUT / "rag_runs"
RUNS.mkdir(parents=True, exist_ok=True)
LOCKED_SHA = "f318984ffec954c373571bbb72f914c718f7db15d50835cb7bbb0799e30b2c71"

ABSTAIN = re.compile(r"bilgis?i? (yok|bulunm|içerm|mevcut değil)|kayıtlarda (bu bilgi )?yok|yanıtlanamaz|bulunmamaktadır|"
                     r"içermez|içermiyor|yer almıyor|belirtilmemiş|bilinmiyor|veremem|söyleyemem|mevcut değil|"
                     r"sahip değil|verilmemiş|ulaşılamıyor|tespit edilemiyor|cevap veremiyorum", re.I)

# Required content for conflict / status questions (written before any run).
MUST = {
    "T06": [["aktif"]],
    "T13": [["29"], ["25"]],
    "T14": [["silivri"], ["büyükçekmece"]],
    "T15": [["99"], ["80"], ["179"]],
    "T16": [["28"], ["95"]],
    "D03": [["aktif"]],
    "D06": [["305"], ["davet"]],
    "D07": [["aktif - pasif", "kesin", "belirsiz", "hem aktif hem pasif"]],
    "D08": [["(aktif / pasif)", "aktif / pasif", "kesin", "belirsiz"]],
}


def numbers(text: str):
    """All numbers in Turkish or English notation (3.834,8 / 3834.8 / 574,2)."""
    text = re.sub(r"GES-\d{4}-\d{3}", " ", text)
    out = []
    for tok in re.findall(r"\d[\d.,]*", text):
        tok = tok.rstrip(".,")
        cands = {tok}
        if "," in tok and "." in tok:
            cands.add(tok.replace(".", "").replace(",", "."))
        elif "," in tok:
            cands.add(tok.replace(",", "."))
        elif tok.count(".") == 1 and len(tok.split(".")[1]) == 3:
            cands.add(tok.replace(".", ""))
        for c in cands:
            try:
                out.append(float(c))
            except ValueError:
                pass
    return out


def cites(text):
    return re.findall(r"GES-\d{4}-\d{3}", text)


def score(q, run, df):
    ans = run["answer"]
    low = ans.lower()
    abst = bool(ABSTAIN.search(ans))
    nums = numbers(ans)
    gold_ids = set(q["evidence_record_ids"])
    if not q["answerable"]:
        correct = abst
    elif q["qid"] in MUST:
        correct = all(any(alt in low for alt in group) for group in MUST[q["qid"]])
    elif q["gold_numeric"] is not None:
        tol = q["tolerance"] or 0.0
        correct = any(abs(n - q["gold_numeric"]) <= tol + 1e-9 for n in nums)
    else:
        correct = False
    numeric_correct = None
    if q["answerable"] and q["gold_numeric"] is not None:
        numeric_correct = any(abs(n - q["gold_numeric"]) <= (q["tolerance"] or 0) + 1e-9 for n in nums)
    c = cites(ans)
    allowed = gold_ids if gold_ids else set(run["retrieved"])
    known = set(df.record_id)
    supp = [x for x in c if x in known and x in allowed]
    # factual numeric claims (exclude file years) supported by gold, tool result or cited records
    support_vals = set()
    if q["gold_numeric"] is not None:
        support_vals.add(round(float(q["gold_numeric"]), 2))
    for rid in set(c) | gold_ids:
        if rid in known:
            r = df.loc[df.record_id == rid].iloc[0]
            support_vals |= {round(float(r.power_kwp), 2), float(r.excel_row)}
    if run.get("tool"):
        support_vals |= {round(n, 2) for n in numbers(json.dumps(run["tool"], ensure_ascii=False))}
    claims = [n for n in nums if not 2020 <= n <= 2030]
    supported_claims = [n for n in claims if round(n, 2) in support_vals]
    recall = None
    if q["category"] == "single" and run["retrieved"]:
        recall = float(bool(gold_ids & set(run["retrieved"])))
    return dict(qid=q["qid"], category=q["category"], answerable=q["answerable"], system=run["system"],
                auto_correct=bool(correct), numeric_correct=numeric_correct, abstained=abst,
                n_citations=len(c), n_supported_citations=len(supp), n_claims=len(claims),
                n_supported_claims=len(supported_claims), recall_at_k=recall, seconds=round(run["seconds"], 2),
                plan_error=run.get("plan_error"), answer=ans.replace("\n", " "))


def ratio(num, den):
    return "uygulanamaz" if den == 0 else round(num / den, 3)


def summarise(sc: pd.DataFrame, correct_col="auto_correct") -> pd.DataFrame:
    rows = []
    for sysname, g in sc.groupby("system"):
        ans_ = g[g.answerable]; un = g[~g.answerable]; num = g[g.numeric_correct.notna()]
        single = g[g.recall_at_k.notna()]
        rows.append({
            "system": sysname, "questions": len(g),
            "accuracy": ratio(g[correct_col].sum(), len(g)),
            "numeric_accuracy": ratio(num.numeric_correct.astype(bool).sum(), len(num)),
            "aggregate_accuracy": ratio(g[g.category == "aggregate"][correct_col].sum(), (g.category == "aggregate").sum()),
            "recall_at_k_single": ratio(single.recall_at_k.sum(), len(single)),
            "citation_accuracy": ratio(g.n_supported_citations.sum(), g.n_citations.sum()),
            "evidence_coverage": ratio(g.n_supported_claims.sum(), g.n_claims.sum()),
            "correct_abstention": ratio(un.abstained.sum(), len(un)),
            "unnecessary_refusal": ratio(ans_.abstained.sum(), len(ans_)),
            "mean_seconds": round(g.seconds.mean(), 2),
        })
    return pd.DataFrame(rows)


def load_gold(split):
    qs = [json.loads(line) for line in open(PROTOCOL / "qa_gold.jsonl", encoding="utf-8")]
    return [q for q in qs if q["split"] == split]


def main(mode):
    from llm import LocalLLM
    from systems import Systems

    df = load_records()
    llm = LocalLLM()
    save_json(llm.info(), OUT_TABLES / "rag_llm_card.json")
    if mode == "dev":
        qs = load_gold("dev")
        res = []
        for k in (3, 5, 10):
            S = Systems(llm, df, k)
            for q in qs:
                run = S.r1(q["question"])
                save_json(run, RUNS / f"dev_R1_k{k}_{q['qid']}.json")
                res.append({**score(q, run, df), "k": k})
        sc = pd.DataFrame(res)
        sc.to_csv(OUT_TABLES / "rag_dev_answers.csv", index=False)
        summ = sc.groupby("k").agg(accuracy=("auto_correct", "mean"), recall=("recall_at_k", "mean"),
                                    citations=("n_citations", "sum"), supported=("n_supported_citations", "sum")).reset_index()
        summ.to_csv(OUT_TABLES / "rag_dev_k_selection.csv", index=False)
        best = summ.sort_values(["accuracy", "recall", "k"], ascending=[False, False, True]).iloc[0]
        save_json({"chosen_k": int(best.k), "rule": "max dev accuracy, then recall, then smaller k"}, OUT_TABLES / "rag_chosen_k.json")
        print(summ.to_string(index=False)); print("chosen k:", int(best.k))
        # development sanity run of all three systems with the chosen k (dev questions only)
        S = Systems(llm, df, int(best.k))
        dres = []
        for q in qs:
            for fn in (S.r0, S.r1, S.r2):
                run = fn(q["question"])
                save_json(run, RUNS / f"dev_{run['system']}_chosen_{q['qid']}.json")
                dres.append(score(q, run, df))
        dsc = pd.DataFrame(dres)
        dsc.to_csv(OUT_TABLES / "rag_dev_all_systems.csv", index=False)
        print(summarise(dsc).to_string(index=False))
    elif mode == "test":
        assert sha256(PROTOCOL / "qa_test_locked.jsonl") == LOCKED_SHA, "locked test set changed!"
        k = json.load(open(OUT_TABLES / "rag_chosen_k.json"))["chosen_k"]
        qs = load_gold("test")
        S = Systems(llm, df, k)
        res = []
        for q in qs:
            for fn in (S.r0, S.r1, S.r2):
                run = fn(q["question"])
                save_json(run, RUNS / f"test_{run['system']}_{q['qid']}.json")
                res.append(score(q, run, df))
        sc = pd.DataFrame(res)
        sc.to_csv(OUT_TABLES / "rag_test_answers.csv", index=False)
        summarise(sc).to_csv(OUT_TABLES / "rag_test_metrics_auto.csv", index=False)
        print(summarise(sc).to_string(index=False))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
