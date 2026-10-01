"""Every prompt the pipeline sends, in one place.

Prompts live client-side so they can change without redeploying the Modal
engine. They deliberately contain no concrete example values: a worked
example such as a real dimension gets copied into answers about drawings
that do not show it.

The two version strings are part of the cache keys, so bumping one redoes
exactly the work that depends on it: `FIGURE_VERSION` for the drawing
reading prompt (minutes of local vision work per document), and
`PROMPT_VERSION` for everything sent to the LLM.
"""
from __future__ import annotations

FIGURE_VERSION = "2026-09-30.5"
PROMPT_VERSION = "2026-09-30.6"

# ---- figures ---------------------------------------------------------------
FIGURE_SYSTEM = (
    "You read drawing sheets from engineering standards (fabrication and "
    "installation details) and list the requirements they impose. You reply "
    "with JSON only.")

FIGURE_USER = """\
This drawing sheet is from the standard "{doc_name}" ({ref}).

Text found on the sheet by OCR (may contain small errors; use it to confirm \
spelling and numbers, and rely on the image for what each label refers to):
{labels}

List every requirement this sheet imposes on design, fabrication or \
installation: one requirement per dimension, member size, material, weld, \
clearance, spacing, limit or note.

Rules:
- Each statement is a complete sentence that stands on its own: it names the \
component, the property and the value.
- Report only what this sheet shows. Never carry over a value from general \
knowledge or from another sheet.
- Keep values exactly as drawn, with both imperial and metric when both appear.
- "detail" is the title of the view or detail the requirement comes from, as \
lettered on the sheet.
- A dimension line that has no wording: report it only if you can name the \
two features it runs between, and say what it measures in those terms. If \
you cannot tell what it measures, leave it out. Do not attach it to a nearby \
label that belongs to something else.
- The same property of the same component appears at most twice (a minimum \
and a maximum).
- A weld symbol (a small flag or triangle on a leader line with a size next \
to it) gives a fillet weld size. It is never a material thickness.
- Skip anything you cannot read. Do not invent values.
- Ignore the title block and cross-references to other pages.

Before the requirements, list the titled views on the sheet. A standard's \
drawings define which arrangements it offers, so this matters as much as \
the dimensions. For each view give its title as lettered, the component it \
is about, and "arrangement": a short noun phrase naming the type of that \
component and how it is configured in this view, specific enough to tell it \
apart from the other types the same component could have.

Return JSON only:
{{"sheet_title": "...", "views": [{{"title": "...", "component": "...", \
"arrangement": "..."}}], "requirements": [{{"detail": "...", "statement": \
"...", "subject": "...", "parameter": "...", "value": "..."}}]}}"""

# ---- requirement extraction ------------------------------------------------
EXTRACT_SYSTEM = """\
You extract requirements from engineering codes and standards for a \
compliance comparison tool. You reply with JSON only.

A requirement is one verifiable statement about design, material, dimension, \
fabrication, installation, inspection, access or safety. Extract EVERY \
requirement in the clause text you are given, one JSON object per atomic \
requirement: one subject, one parameter, one value or condition. Split a \
sentence that contains several.

For each requirement give:
- clause: the reference of the clause it comes from, copied exactly from the \
[...] tag at the start of that clause.
- statement: a complete sentence that stands on its own without the rest of \
the document. Carry in the subject from parent clauses and headings: a list \
item that only gives a value, sitting under a lead-in sentence that names \
what is being specified, becomes one sentence that states both. Keep values \
and units exactly as written. Add nothing the text does not say.
- subject: the component or topic the requirement is about.
- parameter: what is specified about it (a dimension, a material, a spacing, \
whether something must be provided, and so on).
- value: the specified value or condition as written. Empty string if there \
is none.
- modality: "shall", "shall not", "should", "should not" or "may", as \
written. Use "info" for design basis or explanation that is not an obligation.
- quote: the shortest exact excerpt of the source text that supports the \
requirement, copied character for character.
- references: other standards or documents the requirement points to, as a \
list of their designations. Use [] if none.

Skip headings on their own, scope and purpose statements, copyright and \
administrative text, and bare lists of referenced documents.

Return JSON only: {"requirements": [...]}. If the text holds no \
requirements, return {"requirements": []}."""

EXTRACT_USER = """\
Document: {doc_name}
Section: {section}

{block}"""

TABLE_USER = """\
Document: {doc_name}
Section: {section}

The following table{caption} is part of the document. Treat each row or \
limit that imposes something as a requirement; use "{ref}" as the clause.

{table}"""

# ---- judging ---------------------------------------------------------------
_RELATIONS = """\
- "conflict": both specify the same property of the same component and \
cannot both be satisfied as written: different required values, different \
required materials or types, or one prohibits what the other requires or \
shows as its standard detail.
- "a_stricter" or "b_stricter": both specify the same property of the same \
component and can both be satisfied, but one is more restrictive: a tighter \
limit, fewer permitted options, or "shall" where the other says "should" \
for the same thing. Meeting the stricter one satisfies both.
- "equivalent": the same requirement in substance, even if worded \
differently or given in different units."""

_RULES = """\
- Compare values numerically after converting units. An imperial value and \
its rounded metric equivalent are the same value.
- A requirement for one component never conflicts with a requirement for a \
different component, even when both give a value for the same kind of \
property, and even when the two components are attached to each other.
- A requirement marked [drawing] describes the standard detail as drawn. If \
the other document forbids or changes that detail, that is a conflict.
- The type or arrangement of a component is a property like any other. When \
one document limits a component to one type and the other offers a different \
type of that same component as a standard detail, both address the same \
property: the limiting document is the stricter one, and it is a conflict if \
the other type is mandatory.
- A drawing view that merely depicts a component is not equivalent to a \
requirement about that component. "equivalent" needs the same property with \
the same value or meaning on both sides."""

JUDGE_SYSTEM = f"""\
You compare requirements from two engineering standards for a \
code-compliance review. You reply with JSON only.

You are given one requirement from document A and numbered candidate \
requirements from document B that a search engine thought might address the \
same thing. Most candidates will not. For EACH candidate, record these steps \
in order:

- a_about: the physical component the A requirement governs, and the \
property of it that is specified.
- b_about: the same for the candidate.
- same_component: true only if both govern the same physical component.
- same_property: true only if both specify the same property of it.
- relation: one of the following.

{_RELATIONS}
- "related": the same component or topic, but nothing to compare.
- "unrelated": a different subject.

"conflict", "a_stricter", "b_stricter" and "equivalent" are possible only \
when same_component and same_property are both true.

Rules:
{_RULES}

Return JSON only:
{{"results": [{{"candidate": 1, "a_about": "...", "b_about": "...", \
"same_component": true, "same_property": true, "relation": "...", \
"reason": "one sentence naming the values compared"}}]}}
Include every candidate exactly once."""

JUDGE_USER = """\
Document A: {name_a}
A. {req_a}

Document B: {name_b}
{candidates}"""

# ---- review of claimed matches ---------------------------------------------
REVIEW_SYSTEM = f"""\
You review a claimed match between two requirements from different \
engineering standards. The result goes into a compliance report that \
engineers act on, so a wrong conflict costs them time: you are the check \
against that. You reply with JSON only.

You are given both requirements with the clause text and section they come \
from, and the relation that was claimed. Work through these steps in order \
and record each one:

- a_component, b_component: the physical component each requirement governs.
- same_component: true only if they are the same component.
- a_property, b_property: the property of that component each one specifies.
- same_property: true only if they are the same property.
- both_satisfiable: true if a single design can satisfy both as written.
- verdict: one of the following.

{_RELATIONS}
- "not_comparable": different components, different properties or \
different situations, so neither constrains the other.

"conflict", "a_stricter", "b_stricter" and "equivalent" are possible only \
when same_component and same_property are both true.

Rules:
{_RULES}

For "conflict", "a_stricter" and "b_stricter" also give:
- topic: two to five words naming the component and property that differ.
- severity: "high" (safety, structural adequacy, or a change of material or \
section that forces re-fabrication), "medium" (a dimension or detail \
changes) or "low" (minor).
- action: one imperative sentence telling the fabricator what to do. The \
governing document is named in the request and its requirement applies. If \
the non-governing document is the stricter one, say to follow the governing \
document and to confirm the stricter value with the client.

Return JSON only:
{{"a_component": "...", "b_component": "...", "same_component": true, \
"a_property": "...", "b_property": "...", "same_property": true, \
"both_satisfiable": true, "verdict": "...", "topic": "...", \
"severity": "...", "action": "...", "explanation": "one or two sentences \
naming the values compared"}}"""

REVIEW_USER = """\
Governing document: {governing}
Claimed relation: {relation} ({reason})

A: {name_a}
{context_a}
Requirement A: {statement_a}

B: {name_b}
{context_b}
Requirement B: {statement_b}"""

# ---- drawing check ---------------------------------------------------------
VERIFY_SYSTEM = (
    "You check statements that were read from an engineering drawing "
    "against the drawing itself. You reply with JSON only.")

VERIFY_USER = """\
The attached image is a drawing sheet from "{doc_name}" ({ref}).

This statement was read from it:
"{statement}"

Find the corresponding item on the drawing and check the statement against \
what is actually drawn. Answer "confirmed" when the statement is materially \
right even if it leaves out a qualifier or a note; answer "incorrect" only \
when a value, a component or the meaning differs from the drawing.

Return JSON only:
{{"verdict": "confirmed" | "incorrect" | "not_found", "correct_statement": \
"the statement as the drawing actually shows it, or empty if confirmed", \
"evidence": "the label or dimension text on the drawing that you relied on"}}"""

# ---- Ask the Code: library descriptions ------------------------------------
# A description is ADDED beside the original text; it never replaces it.
DESCRIBE_FIGURE_SYSTEM = (
    "You describe figures from engineering codes so they can be found by a "
    "search over the code. You reply with JSON only.")

DESCRIBE_FIGURE_USER = """\
This figure is from "{doc_name}" ({ref}){caption}.

Describe what it shows in two to five sentences: the component or situation, \
what is being defined or dimensioned, the symbols and variables that appear, \
and the cases or options it distinguishes. Transcribe every label, variable \
and value you can read. Do not interpret beyond what is drawn and do not add \
values that are not on the figure.

Return JSON only: {{"description": "..."}}"""

DESCRIBE_TABLE_SYSTEM = (
    "You describe tables from engineering codes so they can be found by a "
    "search over the code. You reply with JSON only.")

DESCRIBE_TABLE_USER = """\
This table is from "{doc_name}", under: {section}.
Caption: {caption}

Columns: {columns}
First rows:
{rows}
{notes}

In two to four sentences say what the table gives (the quantity, its units), \
what it is indexed by (rows and columns), and what it is used for in the code \
if the caption or notes say so. Do not list the values. Return JSON only: \
{{"description": "..."}}"""

# ---- Ask the Code: the answer -----------------------------------------------
ASK_SYSTEM = """\
You answer questions about an engineering code for a piping engineer, using \
ONLY the passages supplied. The passages are tagged [P1], [P2], ... and each \
one names its code, edition, paragraph and page.

Rules:
- Every statement of fact carries the tag of the passage it comes from, like \
[P2]. Do not cite a tag that is not in the list.
- Where the exact wording matters (a limit, a condition, a definition), quote \
it word for word inside double quotes, then cite it.
- If the passages do not answer the question, say so plainly: "Not found in \
the passages supplied" and, if useful, what they do say. Never answer from \
memory of the code.
- Do no arithmetic. If the question needs a calculation, give the equation \
exactly as the passage has it (LaTeX in $$ ... $$ when the passage has it), \
name each variable from the passage, and say which tables or paragraphs \
supply the inputs. Leave the number to the engineer.
- A table value is quoted only if it appears in a passage; otherwise say which \
table holds it.
- Name the code and edition you are answering from in the first sentence, as \
the passages give them.
- Write in plain prose with short paragraphs. Use a bulleted list only for \
genuinely list-like content such as several conditions."""

ASK_USER = """\
Question: {question}

Passages:
{passages}"""
