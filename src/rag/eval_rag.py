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


def manual_report():
    """Final metrics from the hand-written labels (outputs/tables/rag_manual_review.csv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import FIG

    sc = pd.read_csv(OUT_TABLES / "rag_test_answers.csv")
    man = pd.read_csv(OUT_TABLES / "rag_manual_review.csv")
    m = sc.merge(man, on=["qid", "system"], validate="one_to_one")
    m["correct"], m["abstained"] = m.manual_correct.astype(bool), m.manual_abstained.astype(bool)
    final = summarise(m.assign(auto_correct=m.correct), correct_col="correct")
    for cat in ["single", "aggregate", "conflict", "unanswerable"]:
        final[f"acc_{cat}"] = [ratio(m[(m.system == s) & (m.category == cat)].correct.sum(),
                                     ((m.system == s) & (m.category == cat)).sum()) for s in final.system]
    final["auto_vs_manual_agreement"] = [round((m[m.system == s].auto_correct == m[m.system == s].correct).mean(), 3)
                                         for s in final.system]
    final.to_csv(OUT_TABLES / "rag_test_metrics_final.csv", index=False)
    ec = (m[m.error_class.notna() & (m.error_class != "")].groupby(["system", "error_class"]).size()
          .rename("n").reset_index())
    ec.to_csv(OUT_TABLES / "rag_error_classes.csv", index=False)
    # tool-level diagnostics for R2 (was the code result itself right?)
    gold = {q["qid"]: q for q in load_gold("test")}
    tool_rows = []
    for q in gold.values():
        run = json.load(open(RUNS / f"test_R2_{q['qid']}.json", encoding="utf-8"))
        t = run.get("tool")
        res = None if t is None else t.get("result")
        ok = None
        if q["category"] == "aggregate" and q["gold_numeric"] is not None:
            vals = numbers(json.dumps(res, ensure_ascii=False)) if res is not None else []
            ok = any(abs(v - q["gold_numeric"]) <= (q["tolerance"] or 0) + 1e-9 for v in vals)
        elif q["evidence_record_ids"]:
            ok = t is not None and set(q["evidence_record_ids"]) <= set(t.get("record_ids", []))
        elif not q["answerable"]:
            ok = t is not None and (t.get("intent") == "unanswerable" or t.get("n_rows", 0) == 0)
        tool_rows.append(dict(qid=q["qid"], category=q["category"], plan_valid=run.get("plan_error") is None,
                              tool_result_correct=ok, plan_error=run.get("plan_error")))
    tool = pd.DataFrame(tool_rows)
    tool.to_csv(OUT_TABLES / "rag_r2_tool_diagnostics.csv", index=False)

    cats = ["single", "aggregate", "conflict", "unanswerable"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3))
    w = 0.25
    for i, (s, c) in enumerate(zip(["R0", "R1", "R2"], ["#8C8C8C", "#4C72B0", "#55A868"])):
        vals = [m[(m.system == s) & (m.category == k)].correct.mean() for k in cats]
        ax[0].bar(np.arange(4) + (i - 1) * w, vals, w, label=s, color=c)
    ax[0].set_xticks(range(4), [f"{k}\n(n={(m[m.system == 'R0'].category == k).sum()})" for k in cats])
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("accuracy (manual labels)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Locked test set: accuracy by question type", fontsize=9)
    labs = ["accuracy", "correct\nabstention", "unnecessary\nrefusal"]
    for i, (s, c) in enumerate(zip(["R0", "R1", "R2"], ["#8C8C8C", "#4C72B0", "#55A868"])):
        r = final[final.system == s].iloc[0]
        vals = [float(r.accuracy), float(r.correct_abstention), float(r.unnecessary_refusal)]
        ax[1].bar(np.arange(3) + (i - 1) * w, vals, w, label=s, color=c)
    ax[1].set_xticks(range(3), labs); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=7)
    ax[1].set_title("Overall (20 locked questions, Qwen2.5-1.5B-Instruct, k=3)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig33_rag_results.png", bbox_inches="tight"); plt.close(fig)

    agg = tool[tool.category == "aggregate"]
    fig, ax = plt.subplots(figsize=(6, 3))
    stages = ["plan valid", "tool result correct", "final answer correct"]
    vals = [agg.plan_valid.sum(), agg.tool_result_correct.sum(),
            m[(m.system == "R2") & (m.category == "aggregate")].correct.sum()]
    ax.bar(stages, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v}/6", ha="center")
    ax.set_ylim(0, 6.5); ax.set_ylabel("aggregate questions")
    ax.set_title("R2 on aggregate questions: where the pipeline loses answers", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig34_rag_r2_pipeline.png", bbox_inches="tight"); plt.close(fig)
    print(final.to_string(index=False)); print(ec.to_string(index=False)); print(tool.to_string(index=False))


def main(mode):
    if mode == "manual":
        return manual_report()
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
