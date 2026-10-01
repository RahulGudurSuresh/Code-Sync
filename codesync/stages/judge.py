"""Candidate pairs -> relations -> findings.

Two passes, the second narrower and more careful than the first:

  judge    one request per A-requirement with all of its B candidates. The
           model must name the component and property on each side before
           it may call a match, and a match is accepted only if it marked
           both as the same.
  review   every claimed match is re-examined on its own, with the full
           clause text around it, by a prompt whose job is to refute it.

Embedding search returns the nearest requirement whether or not it is
about the same thing, so most candidates are not matches; the passes exist
to keep those out of a report engineers will act on.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from .. import library, prompts
from ..models.chat import chat_json
from ..schema import (DocumentMeta, Element, Finding, ParsedDocument,
                      Requirement)

RELATIONS = {"conflict", "a_stricter", "b_stricter", "equivalent",
             "related", "unrelated"}
MATCHED = {"conflict", "a_stricter", "b_stricter", "equivalent"}
MAX_CANDIDATES = 8
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "": 3}


def _describe(r: Requirement) -> str:
    tag = " [drawing]" if r.from_figure else ""
    text = f"[{r.ref}] ({r.modality}){tag}: {r.statement}"
    if r.quote_verified and r.quote.strip().lower() != r.statement.lower():
        text += f'\n   source text: "{r.quote}"'
    return text


def _true(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def judge_pairs(meta_a: DocumentMeta, meta_b: DocumentMeta,
                reqs_a: list[Requirement], reqs_b: list[Requirement],
                pairs: list[tuple[int, int, float]], llm,
                progress: Callable[[str], None] | None = None
                ) -> tuple[list[dict], dict]:
    """[{a, b, similarity, relation, reason, ...}] for every pair."""
    say = progress or (lambda _m: None)
    by_a: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, j, s in pairs:
        by_a[i].append((j, s))
    groups: list[tuple[int, list[tuple[int, float]]]] = []
    for i, cands in by_a.items():
        cands.sort(key=lambda c: -c[1])
        groups += [(i, cands[k:k + MAX_CANDIDATES])
                   for k in range(0, len(cands), MAX_CANDIDATES)]

    def ask(batch: list[tuple[int, list[tuple[int, float]]]]
            ) -> dict[tuple[int, int], dict]:
        """Verdicts keyed by (index in A, index in B)."""
        requests = []
        for i, chunk in batch:
            listing = "\n".join(f"{n}. {_describe(reqs_b[j])}"
                                for n, (j, _s) in enumerate(chunk, 1))
            requests.append({"messages": [
                {"role": "system", "content": prompts.JUDGE_SYSTEM},
                {"role": "user", "content": prompts.JUDGE_USER.format(
                    name_a=f"{meta_a.label} ({meta_a.name})",
                    req_a=_describe(reqs_a[i]),
                    name_b=f"{meta_b.label} ({meta_b.name})",
                    candidates=listing)}],
                "max_tokens": 300 + 260 * len(chunk)})
        found: dict[tuple[int, int], dict] = {}
        for (i, chunk), (answer, _raw) in zip(batch, chat_json(llm, requests)):
            # the model sometimes returns the list without its wrapper
            results = answer
            if isinstance(answer, dict):
                results = answer.get("results")
            if not isinstance(results, list):
                stats["parse_failed"] += 1
                continue
            for res in results:
                try:
                    n = int(res.get("candidate"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if 1 <= n <= len(chunk):
                    found[(i, chunk[n - 1][0])] = res
        return found

    stats = {"requests": len(groups), "parse_failed": 0, "reasked": 0,
             "unjudged": 0, "gated": 0}
    say(f"judging {len(pairs)} candidate pairs in {len(groups)} requests")
    verdicts = ask(groups) if groups else {}
    # a long candidate list sometimes comes back short: ask again for the
    # ones that are missing, on their own
    missing = [(i, [c for c in chunk if (i, c[0]) not in verdicts])
               for i, chunk in groups]
    missing = [(i, chunk) for i, chunk in missing if chunk]
    if missing:
        stats["reasked"] = sum(len(chunk) for _i, chunk in missing)
        verdicts.update(ask(missing))

    out: list[dict] = []
    for i, chunk in groups:
        for j, s in chunk:
            res = verdicts.get((i, j))
            pair = {"a": reqs_a[i].id, "b": reqs_b[j].id,
                    "similarity": round(s, 4), "relation": "unjudged",
                    "reason": "no verdict returned for this pair"}
            if res is None:
                # unread is not "unrelated": keep it visible as unjudged
                stats["unjudged"] += 1
                out.append(pair)
                continue
            relation = str(res.get("relation", "")).strip().lower()
            if relation not in RELATIONS:
                relation = "related"
            if relation in MATCHED and not (
                    _true(res.get("same_component"))
                    and _true(res.get("same_property"))):
                # the model called a match it had itself marked as being
                # about different things
                relation = "related"
                stats["gated"] += 1
            pair.update(relation=relation,
                        reason=str(res.get("reason", "")).strip(),
                        a_about=str(res.get("a_about", "")).strip(),
                        b_about=str(res.get("b_about", "")).strip())
            out.append(pair)
    return out, stats


def _context(r: Requirement, parsed: ParsedDocument) -> str:
    """Where a requirement sits: its section and the clause text around it."""
    by_id: dict[str, Element] = {e.id: e for e in parsed.elements}
    e = by_id[r.element_id]
    section = " > ".join(e.heading_trail) or "-"
    if r.from_figure:
        return f"Section: {section}\nSource: drawing, {r.ref}"
    chain, p = [e], by_id.get(e.parent) if e.parent else None
    while p is not None and not p.is_heading:
        chain.append(p)
        p = by_id.get(p.parent) if p.parent else None
    text = "\n".join(f"[{c.ref}] {c.text}" for c in reversed(chain))
    return f"Section: {section}\nClause text:\n{text}"


def review_matches(judged: list[dict], meta_a: DocumentMeta,
                   meta_b: DocumentMeta, reqs_a: list[Requirement],
                   reqs_b: list[Requirement], governing: DocumentMeta, llm,
                   progress: Callable[[str], None] | None = None) -> dict:
    """Second look at every claimed match, in place. A pair the reviewer
    finds not comparable is demoted to "related" and leaves the report."""
    say = progress or (lambda _m: None)
    ra = {r.id: r for r in reqs_a}
    rb = {r.id: r for r in reqs_b}
    parsed_a = library.load_parsed(meta_a.doc_id)
    parsed_b = library.load_parsed(meta_b.doc_id)
    claimed = [p for p in judged if p["relation"] in MATCHED]
    stats = {"reviewed": len(claimed), "upheld": 0, "changed": 0,
             "rejected": 0, "unread": 0}
    if not claimed:
        return stats

    requests = [{"messages": [
        {"role": "system", "content": prompts.REVIEW_SYSTEM},
        {"role": "user", "content": prompts.REVIEW_USER.format(
            governing=f"{governing.label} ({governing.name})",
            relation=p["relation"], reason=p["reason"],
            name_a=f"{meta_a.label} ({meta_a.name})",
            context_a=_context(ra[p["a"]], parsed_a),
            statement_a=ra[p["a"]].statement,
            name_b=f"{meta_b.label} ({meta_b.name})",
            context_b=_context(rb[p["b"]], parsed_b),
            statement_b=rb[p["b"]].statement)}],
        "max_tokens": 700} for p in claimed]
    say(f"reviewing {len(claimed)} claimed matches")

    for p, (answer, _raw) in zip(claimed, chat_json(llm, requests)):
        p["first_pass"] = p["relation"]
        if not isinstance(answer, dict):
            stats["unread"] += 1
            p["review"] = "unread"
            continue
        verdict = str(answer.get("verdict", "")).strip().lower()
        same = (_true(answer.get("same_component"))
                and _true(answer.get("same_property")))
        if verdict not in MATCHED or not same:
            stats["rejected"] += 1
            p.update(relation="related", review="rejected",
                     reason=str(answer.get("explanation", "")).strip())
            continue
        stats["upheld" if verdict == p["relation"] else "changed"] += 1
        severity = str(answer.get("severity", "")).strip().lower()
        p.update(
            relation=verdict, review="upheld",
            reason=str(answer.get("explanation", "")).strip() or p["reason"],
            topic=str(answer.get("topic", "")).strip(),
            severity=severity if severity in ("high", "medium", "low") else "",
            action=str(answer.get("action", "")).strip())
    return stats


def _words(text: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " "
                               for c in text).split() if len(w) > 2}


def _merge_repeats(findings: list[Finding]) -> list[Finding]:
    """A standard often states one requirement in several places (a clause
    and a note repeated on three drawing sheets). Each would otherwise be
    its own finding against the same A requirement; keep the best-matching
    one and list the other locations on it."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda f: -(f.similarity or 0.0)):
        twin = next((k for k in kept
                     if k.a.id == f.a.id and k.type == f.type
                     and len(_words(k.b.statement) & _words(f.b.statement))
                     >= 0.6 * len(_words(k.b.statement)
                                  | _words(f.b.statement))), None)
        if twin is None:
            kept.append(f)
        else:
            twin.also_b.append(f"{f.b.ref} (p.{f.b.page})")
            if f.needs_review and not twin.needs_review:
                twin.needs_review, twin.review_reason = True, f.review_reason
    return kept


def build_findings(meta_a: DocumentMeta, meta_b: DocumentMeta,
                   reqs_a: list[Requirement], reqs_b: list[Requirement],
                   judged: list[dict], governing: DocumentMeta
                   ) -> list[Finding]:
    ra = {r.id: r for r in reqs_a}
    rb = {r.id: r for r in reqs_b}
    other = meta_b if governing.doc_id == meta_a.doc_id else meta_a
    findings: list[Finding] = []
    matched_a: set[str] = set()
    matched_b: set[str] = set()

    for p in judged:
        rel = p["relation"]
        if rel not in MATCHED:
            continue
        matched_a.add(p["a"])
        matched_b.add(p["b"])
        f = Finding(id="", category="similar", type="equivalent",
                    topic=p.get("topic", ""), a=ra[p["a"]], b=rb[p["b"]],
                    similarity=p["similarity"], rationale=p["reason"])
        if rel != "equivalent":
            f.category = "conflict"
            f.type = "contradiction" if rel == "conflict" else "more_stringent"
            f.governing = governing.label
            f.action = p.get("action", "")
            f.severity = p.get("severity") or "medium"
            if rel in ("a_stricter", "b_stricter"):
                stricter = meta_a if rel == "a_stricter" else meta_b
                f.stricter = stricter.label
                if stricter.doc_id == other.doc_id:
                    f.needs_review = True
                    f.review_reason = (
                        f"{other.label} is stricter than the governing "
                        f"document; confirm which value applies")
        if f.a.from_figure or f.b.from_figure:
            # only statements both readings agreed on get this far
            f.drawing_check = "confirmed"
        if p.get("review") == "unread":
            f.needs_review = True
            f.review_reason = "; ".join(x for x in (
                f.review_reason, "the review pass returned no answer") if x)
        findings.append(f)

    findings = _merge_repeats(findings)

    for r in reqs_a:
        if r.id not in matched_a:
            findings.append(Finding(id="", category="unique",
                                    type="only_in_a", topic=r.subject, a=r))
    for r in reqs_b:
        if r.id not in matched_b:
            findings.append(Finding(id="", category="unique",
                                    type="only_in_b", topic=r.subject, b=r))

    rank = {"contradiction": 0, "more_stringent": 1, "equivalent": 2,
            "only_in_a": 3, "only_in_b": 4}
    findings.sort(key=lambda f: (rank[f.type], _SEVERITY_ORDER[f.severity],
                                 -(f.similarity or 0.0)))
    for n, f in enumerate(findings, 1):
        f.id = f"F{n:04d}"
    return findings
