# CodeSync Analyzer

Compare two engineering codes and report where they conflict, agree or
differ; ask a question of an indexed code and get an answer with the
paragraph and page cited. The backend and web app live in `codesync/`,
`modal_app/` and `frontend/`; see [BACKEND.md](BACKEND.md) for how it works,
how to run it and what was measured, and [BOTTLENECKS.md](BOTTLENECKS.md) for
the known risks.

**The PDFs are not in this repository** (ASME and PIP standards are
copyrighted; the BP documents are a client's internal specifications), and
neither is `data/`, which is derived from them. To run the demo, put the
PDFs under `codes/` as listed below and in `codes/test_codes/`, then follow
the commands in BACKEND.md. The rest of this file is about the reference
codes in `codes/`.

Started 2026-09-28. Purpose of this folder: get real piping design codes in hand,
see how they are built and how they differ, before designing any compliance-checking
automation.

## What is in `codes/`

| File | What it is | Pages | Edition | Source (legitimate copy) |
|---|---|---|---|---|
| `ASME_B31.1-2007_Power_Piping.pdf` | Full ASME B31.1 Power Piping code | 338 | 2007 | Public.Resource.Org scan on the Internet Archive: https://archive.org/details/gov.law.asme.b31.1.2007 |
| `ASME_B31.8-2003_Gas_Transmission_Piping.pdf` | Full ASME B31.8 Gas Transmission and Distribution Piping Systems | 194 | 2003 | Public.Resource.Org: https://law.resource.org/pub/us/cfr/ibr/002/asme.b31.8.2003.pdf |
| `DOE_Comparison_EN13480_vs_ASME_B31.3.pdf` | "Assessment of Safety Equivalency, EN 13480 – ASME B31.3", George Antaki, PRE-L-02, Aug 2024 | 59 | compares B31.3-2022 vs EN 13480-2017 | US Department of Energy: https://www.energy.gov/sites/default/files/2024-08/Report%20-%20EN%2013480,%20ASME%20B31%20Comparison%20-%2021%20A.pdf |
| `text/*.txt` | `pdftotext -layout` extracts of the three PDFs, grep-able | | | generated locally |

Legal note. B31.1 and B31.8 are ASME copyright. These editions were incorporated by
reference into US federal regulations, and Public.Resource.Org posts such standards
non-commercially. The DC Circuit held that posting to be fair use (ASTM v.
Public.Resource.Org, 2023). Fine for study and for building a tool against; do not
redistribute the PDFs. The DOE report is a US government work.

The archive.org copies are scans with OCR, so the text extracts contain artefacts
("VVeld" for "Weld", "\-velded" for "welded"). Equations are mangled in the text; read
those from the PDF.

## Why these two codes

The two codes you most likely care about for EN-based process piping are ASME B31.3
(Process Piping) and EN 13480-3. Both are paywalled and no legitimate free copy exists.
So:

- **B31.1 stands in for B31.3.** It is B31.3's sibling in the B31 family. Same chapter
  skeleton, same paragraph numbering scheme (1xx in B31.1, 3xx in B31.3), same
  allowable-stress design philosophy, same Appendix A stress tables. Learn B31.1's
  structure and you know how B31.3 is laid out.
- **B31.8 is the contrast.** It is also ASME B31, but built on a completely different
  design philosophy (yield strength times a design factor chosen by population density).
  Reading it next to B31.1 shows how different two "piping codes" can be.
- **The DOE report is the bridge to EN 13480.** It walks attribute by attribute through
  B31.3 vs EN 13480 (materials, design, fabrication, examination, testing, overpressure
  protection) and quotes the relevant clauses of both, so you get the EN side without the
  EN document.

## How the two codes are built

| | ASME B31.1-2007 Power Piping | ASME B31.8-2003 Gas Transmission |
|---|---|---|
| Ch. I | Scope and Definitions | Materials and Equipment |
| Ch. II | Design (paras. 101 loads, 102 design criteria, 104 pressure design of components) | Welding |
| Ch. III | Materials | Piping System Components and Fabrication Details |
| Ch. IV | Dimensional Requirements | Design, Installation, and Testing (paras. 840 to 849) |
| Ch. V | Fabrication, Assembly, and Erection | Operating and Maintenance Procedures |
| Ch. VI | Inspection, Examination, and Testing | Corrosion Control |
| Ch. VII | Operation and Maintenance | Miscellaneous |
| Ch. VIII | | Offshore Gas Transmission |
| Appendices | A = Allowable Stress Tables (the heart of the code), B to J | A to Q (incl. Appendix D SMYS of common steels) |
| Para numbering | 1xx (B31.3 uses 3xx, B31.4 uses 4xx, B31.8 uses 8xx) | 8xx |

Both codes are "construction" codes for new piping, but B31.8 spends more than half its
pages on operation, maintenance, corrosion control and integrity, because a pipeline is
managed for decades. B31.1 only gained an O&M chapter (VII) in 2007.

## How they differ in design philosophy

| Aspect | B31.1 (and B31.3) | B31.8 |
|---|---|---|
| Straight pipe wall thickness | para. 104.1.2, eq. (3): `tm = P*Do / (2*(SE + P*y)) + A` | para. 841.11: `P = 2*S*t/D * F * E * T` |
| Strength basis | Allowable stress S at design temperature, read from Appendix A tables. Basis is BPVC Section II-D Appendix 1 (para. 102.3.1(C)): roughly tensile/3.5 and 2/3 of yield, plus creep limits | Specified minimum yield strength (SMYS) of the pipe grade, straight from the material spec (API 5L etc.) |
| Safety margin lives in | The S table (material by material, temperature by temperature) | The design factor F, Table 841.114A: 0.80 (Loc. Class 1 Div. 1), 0.72 (Class 1 Div. 2), 0.60 (Class 2), 0.50 (Class 3), 0.40 (Class 4) |
| Risk grading | None from population. Service categories exist only via boiler external piping rules | Location Class 1 to 4 (para. 840.2), driven by buildings for human occupancy per mile. Higher class = thicker pipe and more weld radiography (10% / 15% / 40% / 75%) |
| Temperature effect | Inside the S(T) table, plus the y coefficient (Table 104.1.2(A): 0.4 for ferritic steels to 900 F, rising to 0.7 by 1000 F) | Temperature derating factor T, Table 841.116A: 1.000 up to 250 F, 0.967 at 300 F, 0.900 at 400 F, 0.867 at 450 F. Code is not meant for hot service |
| Corrosion and tolerances | Explicit: A adds corrosion, threading, mechanical allowances; then add the mill under-tolerance (para. 104.1.2(A1.1)) and round up to a commercial schedule | Mill under-tolerance is absorbed in F (statement under 841.11); corrosion is handled by Chapter VI corrosion control, not by adding thickness |
| Weld joint factor | E, in the tabulated SE values | E, Table 841.115A / 841.114B |
| Flexibility and sustained stress | Paras. 104.8 and 119: sustained, occasional and displacement stress limits, stress range SA = f*(1.25 Sc + 0.25 Sh) | Para. 833: combined stress limits as a percentage of SMYS |

The practical difference for a checker: a B31.1/B31.3 check needs a material database
(S versus temperature, y, E) and an allowance policy. A B31.8 check needs a location
class survey and a SMYS lookup. Same physical formula (Barlow), different inputs.

## The EN 13480 vs B31.3 bridge (from the DOE report)

Verified against the report text:

- **Scope floor.** B31.3 excludes piping below 15 psi in benign service (para. 300.1.3).
  The PED, and hence EN 13480, applies above 0.5 bar (7.25 psi), about half that.
- **Structure.** EN 13480 is eight separate parts: 1 General, 2 Materials, 3 Design and
  calculation, 4 Fabrication and installation, 5 Inspection and testing, 6 Buried piping,
  7 Conformity assessment, 8 Aluminium. B31.3 is one book with the same topics as
  chapters.
- **Legal position.** B31.3 is a voluntary standard adopted by owners or regulators.
  EN 13480 is a harmonised standard under the Pressure Equipment Directive 2014/68/EU,
  which is mandatory law in the EU. Compliance with EN 13480 is presumed to satisfy the
  PED's Essential Safety Requirements; equipment is graded into hazard categories 0 to IV
  and CE marked.
- **Allowable stress below creep** (report section 6.5):
  - B31.3 para. 302.3.2: S = min(1/3 tensile, 2/3 yield); austenitic: min(1/3 tensile, 90% yield).
  - EN 13480-3 para. 5.2.1.1: f = min(1/2.4 tensile, 2/3 yield); austenitic with rupture
    elongation over 35%: min(1/2.4 tensile, 1/1.2 yield).
  - So EN allows a higher fraction of tensile strength (1/2.4 instead of 1/3) but takes
    the same 2/3 of yield, which usually governs for carbon steel at moderate temperature.
  - Creep: EN 13480 uses 200,000 h rupture data with a 1.25 or 1.5 factor, B31.3 uses
    100,000 h data with a 0.67 factor.
- **Weld examination** (section 8.2): both graded. B31.3 grades by Fluid Service
  (Normal = visual + 5% RT, Category M / severe cyclic / high pressure = 100% RT,
  Category D = visual only). EN 13480-5 Table 8.2-1 grades by material group and PED
  piping category I to III, 5% to 100% volumetric.
- **Overall conclusion** (section 3): EN 13480 with the PED gives an equivalent or more
  restrictive level of safety than B31.3, except for seven attributes where B31.3 is
  stricter: burst-test qualification margin of 3 for unlisted fittings, bolted flange
  restrictions (slip-on flanges double-welded in toxic/flammable/cryogenic service, weld
  neck under severe cyclic conditions, no low-yield bolting under severe cyclic
  conditions), brazing and soldering not covered by EN 13480-4, pressure testing of
  externally pressured (vacuum) piping, testing of double-wall piping, over-pressure
  protection, and leak test at startup.

## Other legitimately free codes, if wanted later

Same Public.Resource.Org collection (not downloaded):

- ASME B31.4-2002 Liquid Pipelines: https://law.resource.org/pub/us/cfr/ibr/002/asme.b31.4.2002.pdf
- ASME B31.5-2001 Refrigeration Piping: https://archive.org/details/gov.law.asme.b31.5.2001
- ASME B31.9-2008 Building Services Piping: https://archive.org/details/gov.law.asme.b31.9.2008
- ASME B31G-1991 Remaining Strength of Corroded Pipelines: https://law.resource.org/pub/us/cfr/ibr/002/asme.b31g.1991.pdf
- ASME B31.8S-2004 Gas Pipeline Integrity: https://law.resource.org/pub/us/cfr/ibr/002/asme.b31.8s.2004.pdf
- IS 2825:1969 Code for Unfired Pressure Vessels (BIS, Indian style): https://archive.org/details/gov.in.is.2825.1969
- ASME 1943 Code for Unfired Pressure Vessels (historic): https://law.resource.org/pub/us/cfr/ibr/002/asme.upv.1943.pdf

Not free anywhere legitimate: ASME B31.3, EN 13480 (any part), ASME BPVC Section VIII,
EN 13445. Scribd and similar uploads of these are pirated copies.

Indian Boiler Regulations 1950 (a statutory piping/boiler code with its own thickness
formulas in Reg. 350) is published free by state boiler directorates, for example
https://dfb.odisha.gov.in/indian-boiler-regulation-1950 , if an Indian statutory code is
ever needed alongside the ASME ones.
