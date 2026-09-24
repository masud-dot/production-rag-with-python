"""Phase 2.5 experiment runner. Every number printed is executed."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from lighthouse._compat import (Candidate, FixedChunker, FlatStore, KeywordIndex,
                      LsaEmbedder, ParentChunker, RecursiveChunker,
                      SpacyEmbedder, StructuralChunker, UserContext,
                      allowed_ids,
                      build_chunks, hit_rate, ndcg_at_k, precision_at_k,
                      recall_at_k, reciprocal_rank, rrf)

ROOT = Path(__file__).resolve().parents[1]
import os
EMB_CLS = SpacyEmbedder if os.environ.get('EMB', 'spacy') == 'spacy' else LsaEmbedder
SEED = 20260916


def load():
    docs = json.loads((ROOT / "data" / "corpus.json").read_text())
    facts = json.loads((ROOT / "data" / "facts.json").read_text())
    return docs, facts


# ------------------------------------------------------------ golden set
def build_golden(docs, facts):
    """Questions whose gold chunks are known by construction."""
    rng = random.Random(SEED)
    by_id = {d["doc_id"]: d for d in docs}
    ends = [d for d in facts]
    examples = []

    def add(q, doc_ids, segment, answerable=True, ctx_role=None):
        examples.append({"question": q, "gold_docs": sorted(doc_ids),
                         "segment": segment, "answerable": answerable,
                         "role": ctx_role or "claims_handler"})

    # identifier: exact endorsement lookups (26)
    for did in rng.sample(ends, 26):
        f = facts[did]
        add(f"What is the {f['peril']} sub-limit under endorsement {did}?",
            [did], "identifier")
    # rare term: subrogation / consequential loss phrasing (13)
    rare = [d for d in docs if "consequential loss" in d["text"]]
    for d in rng.sample(rare, 13):
        add("Which clause excludes consequential loss for this wording "
            f"in {d['title']}?", [d["doc_id"]], "rare_term")
    # constraint: superseded / jurisdiction (10)
    sup = [d for d in docs if d["effective_to"]][:10]
    for d in sup:
        add(f"What is the current position replacing {d['doc_id']}?",
            [d["doc_id"]], "constraint")
    # similar-not-relevant: flood vs escape of water (6)
    eow = [d for d in docs if "escape of water" in d["text"]
           and d["collection"] == "wordings"]
    for d in rng.sample(eow, 6):
        add("What is the escape of water sub-limit, not the flood "
            f"sub-limit, in {d['title']}?", [d["doc_id"]], "similar")
    # conceptual (145)
    pool = [d for d in docs if d["collection"] in
            ("manuals", "faqs", "bulletins", "wordings")]
    for d in rng.sample(pool, 145):
        if d["collection"] == "manuals":
            q = "When must a claims handler escalate to a senior handler?"
        elif d["collection"] == "faqs":
            q = f"How is damage assessed for {d['title']}?"
        elif d["collection"] == "bulletins":
            q = "What record keeping do firms need for pricing decisions?"
        else:
            q = f"What cover applies under {d['title']}?"
        add(q, [d["doc_id"]], "conceptual")
    # restricted probes (8) - answerable only for underwriter
    sops = [d for d in docs if d["access"] == "restricted"][:8]
    for d in sops:
        add(f"What loading applies under pricing memorandum {d['doc_id']}?",
            [d["doc_id"]], "restricted_probe", ctx_role="broker_support")
    return examples


def golden_grades(examples, metas):
    """Map gold documents to graded chunk ids: 2 for gold doc chunks."""
    by_doc = defaultdict(list)
    for m in metas:
        by_doc[m.doc_id].append(m.chunk_id)
    out = []
    for ex in examples:
        grades = {}
        for did in ex["gold_docs"]:
            for cid in by_doc.get(did, []):
                grades[cid] = 2
        out.append({**ex, "grades": grades})
    return out


# ------------------------------------------------------------ experiment
def run_config(store, keyword, embedder, gold, mode, k=5, depth=50,
               ctx=None, metas=None):
    rows = defaultdict(lambda: defaultdict(list))
    allow = allowed_ids(metas, ctx) if ctx else None
    for ex in gold:
        gset = set(ex["grades"])
        if mode == "dense":
            cands = store.search(embedder.encode([ex["question"]])[0],
                                 k, allow)
        elif mode == "bm25":
            cands = keyword.search(ex["question"], k, allow)
        else:
            d = store.search(embedder.encode([ex["question"]])[0],
                             depth, allow)
            b = keyword.search(ex["question"], depth, allow)
            cands = rrf([d, b])[:k]
        ranked = [c.meta.chunk_id for c in cands]
        seg = ex["segment"]
        rows[seg]["hit"].append(hit_rate(ranked, gset, k))
        rows[seg]["recall"].append(recall_at_k(ranked, gset, k))
        rows[seg]["mrr"].append(reciprocal_rank(ranked, gset))
        rows[seg]["ndcg"].append(ndcg_at_k(ranked, ex["grades"], k))
        rows["ALL"]["hit"].append(hit_rate(ranked, gset, k))
        rows["ALL"]["recall"].append(recall_at_k(ranked, gset, k))
        rows["ALL"]["mrr"].append(reciprocal_rank(ranked, gset))
        rows["ALL"]["ndcg"].append(ndcg_at_k(ranked, ex["grades"], k))
    return {s: {m: round(float(np.mean(v)), 3) for m, v in d.items()}
            for s, d in rows.items()}


def main():
    docs, facts = load()
    print(f'embedder: {EMB_CLS.__name__}')
    print(f"corpus: {len(docs)} documents\n")

    # ---- Chapter 6: chunking comparison -----------------------------
    print("== CH6 chunking comparison (hit rate @5, dense LSA) ==")
    chunk_results = {}
    stores = {}
    for chunker in (FixedChunker(), RecursiveChunker(),
                    StructuralChunker(), ParentChunker()):
        metas, texts = build_chunks(docs, chunker)
        emb = EMB_CLS()
        vecs = emb.fit_transform(texts)
        store = FlatStore(metas, texts, np.asarray(vecs))
        kw = KeywordIndex(metas, texts)
        gold = golden_grades(build_golden(docs, facts), metas)
        res = run_config(store, kw, emb, gold, "dense", metas=metas)
        chunk_results[chunker.name] = {
            "chunks": len(metas),
            "overall_hit": res["ALL"]["hit"],
            "by_seg": {s: res[s]["hit"] for s in res if s != "ALL"}}
        stores[chunker.name] = (metas, texts, emb, store, kw, gold)
        print(f"  {chunker.name:11s} chunks={len(metas):6d} "
              f"hit@5={res['ALL']['hit']:.3f}")
    (ROOT / "evaluation" / "results" / "ch6_chunking.json").write_text(
        json.dumps(chunk_results, indent=2))

    # ---- Chapters 11/12/17/21: retrieval modes on structural --------
    metas, texts, emb, store, kw, gold = stores["structural"]
    print(f"\n== CH11/12/17 retrieval modes (structural, "
          f"{len(metas)} chunks) ==")
    modes = {}
    for mode in ("dense", "bm25", "hybrid"):
        res = run_config(store, kw, emb, gold, mode, metas=metas)
        modes[mode] = res
        a = res["ALL"]
        print(f"  {mode:7s} hit={a['hit']:.3f} recall@5={a['recall']:.3f} "
              f"mrr={a['mrr']:.3f} ndcg={a['ndcg']:.3f}")
    print("\n  per-segment hit rate:")
    segs = [s for s in modes["dense"] if s != "ALL"]
    print("  " + "segment".ljust(18)
          + "".join(m.ljust(10) for m in modes))
    for s in segs:
        n = sum(1 for g in gold if g["segment"] == s)
        print(f"  {s:<18}"
              + "".join(f"{modes[m][s]['hit']:<10.3f}" for m in modes)
              + f"(n={n})")
    (ROOT / "evaluation" / "results" / "ch12_modes.json").write_text(
        json.dumps(modes, indent=2))

    # ---- Chapter 13: filtering + leakage ----------------------------
    print("\n== CH13 filtering and leakage ==")
    broker = UserContext(role="broker_support")
    under = UserContext(role="underwriter")
    probes = [g for g in gold if g["segment"] == "restricted_probe"]
    leak_unfiltered = 0
    leak_filtered = 0
    allow_b = allowed_ids(metas, broker)
    for ex in probes:
        gset = set(ex["grades"])
        un = store.search(emb.encode([ex["question"]])[0], 5, None)
        fi = store.search(emb.encode([ex["question"]])[0], 5, allow_b)
        leak_unfiltered += int(bool({c.meta.chunk_id for c in un} & gset))
        leak_filtered += int(bool({c.meta.chunk_id for c in fi} & gset))
    print(f"  restricted probes            : {len(probes)}")
    print(f"  leakage, no filter           : {leak_unfiltered}/{len(probes)}")
    print(f"  leakage, broker_support filt : {leak_filtered}/{len(probes)}")
    print(f"  allowed chunks broker        : {len(allow_b)}/{len(metas)} "
          f"({len(allow_b) / len(metas):.1%})")
    print(f"  allowed chunks underwriter   : "
          f"{len(allowed_ids(metas, under))}/{len(metas)}")

    # ---- Chapter 17: metric formula verification --------------------
    print("\n== CH17 metric formulas (worked example from exercise) ==")
    # Figure 17.1: one relevant chunk (G, grade 1) sits at rank 8,
    # outside the cut. Verified in Phase 2.6.
    ranked = ["A", "B", "C", "D", "E"]
    grades = {"B": 2, "D": 2, "G": 1}
    gset = set(grades)
    print(f"  hit@5       = {hit_rate(ranked, gset, 5):.3f}")
    print(f"  recall@5    = {recall_at_k(ranked, gset, 5):.3f}")
    print(f"  precision@5 = {precision_at_k(ranked, gset, 5):.3f}")
    print(f"  MRR         = {reciprocal_rank(ranked, gset):.3f}")
    print(f"  nDCG@5      = {ndcg_at_k(ranked, grades, 5):.3f}")
    reord = ["B", "D", "A", "G", "C"]
    print(f"  after rerank -> MRR {reciprocal_rank(reord, gset):.3f}, "
          f"nDCG {ndcg_at_k(reord, grades, 5):.3f}")

    # ---- Chapters 7/8/24: deterministic calculations ----------------
    print("\n== CH7/8/24 deterministic calculations ==")
    n_chunks = len(metas)
    avg_chars = sum(len(t) for t in texts) / n_chunks
    tokens = sum(len(t) for t in texts) / 4
    print(f"  chunks (structural)      : {n_chunks}")
    print(f"  mean chunk chars         : {avg_chars:.0f}")
    print(f"  est. corpus tokens (/4)  : {tokens:,.0f}")
    for dim in (768, 1536, 3072):
        for n in (n_chunks, 312_481):
            vec = n * dim * 4 / 1e9
            graph = n * 32 * 2 * 4 / 1e9
            if dim == 1536:
                print(f"  n={n:>7,} dim={dim}: vectors {vec:.2f} GB "
                      f"+ graph {graph:.2f} GB = {vec + graph:.2f} GB")
    print("\n  Ch24 per-query cost model (1200+180 in, 180 out):")
    for label, rin, rout in (("strong", 2.20, 8.80), ("fast", 0.35, 1.40)):
        c = (1380 * rin + 180 * rout) / 1e6
        print(f"    {label:6s} GBP {c:.6f}/query, "
              f"GBP {c * 50000:.2f}/month at 50k")

    json.dump({"chunk_results": chunk_results, "modes": modes,
               "leakage": {"unfiltered": leak_unfiltered,
                           "filtered": leak_filtered,
                           "probes": len(probes)}},
              open(ROOT / "evaluation" / "results" / "all_results.json", "w"), indent=2)
    print("\nwrote evaluation/results/all_results.json")


if __name__ == "__main__":
    main()
