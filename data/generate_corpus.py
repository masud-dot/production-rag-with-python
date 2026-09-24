"""Deterministic synthetic Harbourline Insurance corpus."""
from __future__ import annotations
import hashlib
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

SEED = 20260916
PLACES = ["Hartfield", "Dunwich", "Kelvedon", "Aberlour", "Pentre",
          "Ballyclare", "Ravenglass", "Thirsk", "Wymondham", "Corfe",
          "Ilkley", "Malvern", "Penrhyn", "Sedgemoor", "Tarbert"]
TRADES = ["bakery", "warehouse", "print works", "cold store", "joinery",
          "dairy", "foundry", "call centre", "garden centre", "brewery"]
PERILS = ["flood", "escape of water", "subsidence", "fire",
          "storm", "theft", "accidental damage", "impact",
          "riot", "business interruption"]
PRODUCTS = ["Mid-Market Commercial Property", "Retail Package",
            "Contractors All Risks", "Motor Fleet",
            "Professional Indemnity", "Marine Cargo"]
JURIS = ["GB", "IE"]


def _d(rng, start=2016, end=2026):
    return date(rng.randint(start, end), rng.randint(1, 12),
                rng.randint(1, 28))


def wording(rng, n):
    prod = rng.choice(PRODUCTS)
    doc_id = f"WRD-{n:04d}"
    blocks = []
    for section in range(1, rng.randint(5, 9)):
        head = rng.choice(PERILS)
        blocks.append({"page": section, "clause": f"{section}",
                       "text": f"Section {section} - {head.title()} Cover"})
        for sub in range(1, rng.randint(3, 6)):
            peril = rng.choice(PERILS)
            limit = rng.choice([10, 25, 50, 100, 250]) * 1000
            place = rng.choice(PLACES)
            trade = rng.choice(TRADES)
            blocks.append({
                "page": section, "clause": f"{section}.{sub}",
                "text": (f"{section}.{sub} The sub-limit for {peril} at the "
                         f"{place} {trade} under this {prod} policy is GBP "
                         f"{limit:,}, reference {doc_id}-{section}{sub}. "
                         f"Cover applies where the {trade} premises were "
                         f"occupied within the preceding thirty days. This "
                         f"clause does not extend to consequential loss "
                         f"arising at {place}.")})
    return doc_id, prod, blocks


def endorsement(rng, n):
    doc_id = f"END-{4000 + n}-{rng.choice('ABC')}"
    peril = rng.choice(PERILS)
    place = rng.choice(PLACES)
    trade = rng.choice(TRADES)
    limit = rng.choice([10, 25, 50, 100]) * 1000
    blocks = [
        {"page": 1, "clause": "1",
         "text": f"Endorsement {doc_id} - {peril.title()}"},
        {"page": 1, "clause": "4.2",
         "text": (f"4.2 The {peril} sub-limit at the {place} {trade} is "
                  f"amended to GBP {limit:,}. No waiting period applies. "
                  f"This endorsement is issued under endorsement reference "
                  f"{doc_id} and takes effect from the date in the "
                  f"schedule for the {place} risk.")},
        {"page": 1, "clause": "4.3",
         "text": (f"4.3 Where the {place} insured has notified a prior "
                  f"{peril} claim, the excess is increased by GBP 2,500 "
                  f"for the remainder of the period of insurance.")},
    ]
    return doc_id, peril, limit, blocks


def manual(rng, n):
    doc_id = f"MAN-{n:03d}"
    blocks = []
    for step in range(1, rng.randint(6, 11)):
        blocks.append({"page": 1 + step // 4, "clause": f"{step}",
                       "text": (f"{step}. On receipt of a "
                                f"{rng.choice(PERILS)} claim the handler "
                                f"confirms cover, records the reserve, and "
                                f"refers to a senior handler where the "
                                f"estimate exceeds GBP 50,000. Escalation "
                                f"to the technical team is required for "
                                f"disputed liability.")})
    return doc_id, blocks


def bulletin(rng, n):
    doc_id = f"BUL-{n:03d}"
    return doc_id, [{"page": 1, "clause": None,
                     "text": (f"Regulatory bulletin {doc_id}. Firms writing "
                              f"{rng.choice(PRODUCTS)} business must record "
                              f"the basis of {rng.choice(PERILS)} pricing "
                              f"decisions and retain evidence for six "
                              f"years.")}]


def faq(rng, n, peril):
    doc_id = f"FAQ-{n:03d}"
    place = rng.choice(PLACES)
    return doc_id, [{"page": 1, "clause": None,
                     "text": (f"Q: How is {peril} damage assessed at a "
                              f"{rng.choice(TRADES)}? A: "
                              f"{peril.capitalize()} damage is assessed by "
                              f"an appointed loss adjuster in {place}. The "
                              f"sub-limit in the schedule applies. Contact "
                              f"the {place} claims team for guidance.")}]


def sop(rng, n):
    doc_id = f"SOP-{n:03d}"
    return doc_id, [{"page": 1, "clause": "1",
                     "text": (f"Internal pricing memorandum {doc_id}. The "
                              f"loading applied to {rng.choice(PERILS)} "
                              f"exposure is {rng.randint(3, 18)} per cent of "
                              f"base rate. This document is restricted to "
                              f"underwriting staff.")}]


def build():
    rng = random.Random(SEED)
    docs = []
    facts = {}

    def add(doc_id, collection, blocks, access="internal", eff=None,
            eff_to=None, sup=None, juris="GB", title=""):
        text = "\n".join(b["text"] for b in blocks)
        docs.append({
            "doc_id": doc_id, "collection": collection,
            "title": title or doc_id, "blocks": blocks, "text": text,
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "effective_from": (eff or _d(rng)).isoformat(),
            "effective_to": eff_to.isoformat() if eff_to else None,
            "supersedes": sup, "jurisdiction": juris, "access": access,
            "loader_version": "1.0.0"})

    for n in range(1, 241):
        did, prod, blocks = wording(rng, n)
        add(did, "wordings", blocks, title=f"{prod} Policy Wording")
    prev = None
    for n in range(1, 361):
        did, peril, limit, blocks = endorsement(rng, n)
        superseded = rng.random() < 0.22
        eff = _d(rng, 2019, 2026)
        add(did, "wordings", blocks, eff=eff,
            eff_to=(eff + timedelta(days=rng.randint(200, 900)))
            if superseded else None,
            sup=prev if rng.random() < 0.3 else None,
            juris=rng.choice(JURIS) if rng.random() < 0.12 else "GB",
            title=f"Endorsement {did}")
        facts[did] = {"peril": peril, "limit": limit,
                      "superseded": superseded}
        prev = did
    for n in range(1, 121):
        did, blocks = manual(rng, n)
        add(did, "manuals", blocks, title=f"Claims Handling Manual {n}")
    for n in range(1, 161):
        did, blocks = bulletin(rng, n)
        add(did, "bulletins", blocks, access="public",
            juris=rng.choice(JURIS) if rng.random() < 0.3 else "GB",
            title=f"Bulletin {did}")
    for n in range(1, 241):
        did, blocks = faq(rng, n, PERILS[n % len(PERILS)])
        add(did, "faqs", blocks, access="public", title=f"FAQ {did}")
    for n in range(1, 81):
        did, blocks = sop(rng, n)
        add(did, "sops", blocks, access="restricted",
            title=f"Pricing Memo {did}")
    return docs, facts


if __name__ == "__main__":
    docs, facts = build()
    root = Path(__file__).resolve().parents[1] / "data"
    (root / "corpus.json").write_text(json.dumps(docs))
    (root / "facts.json").write_text(json.dumps(facts))
    h = hashlib.sha256((root / "corpus.json").read_bytes()).hexdigest()[:16]
    print(f"documents          : {len(docs)}")
    print("by collection      :",
          dict(Counter(d["collection"] for d in docs)))
    print("by access          :", dict(Counter(d["access"] for d in docs)))
    print("superseded         :",
          sum(1 for d in docs if d["effective_to"]))
    print("IE jurisdiction    :",
          sum(1 for d in docs if d["jurisdiction"] == "IE"))
    print("total characters   :", sum(len(d["text"]) for d in docs))
    print(f"corpus_hash        : {h}")
