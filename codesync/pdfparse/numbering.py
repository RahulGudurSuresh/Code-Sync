"""Rows -> clause tree.

Codes nest requirements several levels deep ("11.6 > a. > 3. > b) > ii)"),
and a leaf such as "ii) 40 in. for passageway" means nothing without its
ancestors. The tree is rebuilt from two signals:

  * the enumerator sequence — "b)" continues an open "a)" level, "i)" after
    "h)" is the letter i while "i)" anywhere else opens a roman level;
  * indentation, which settles where un-numbered paragraphs belong.

Sequence comes first because it survives layouts that do not indent (ASME
prints every level flush left); indentation is the tie-breaker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lines import Row, is_toc_row

BULLETS = {"•", "–", "-", "—", "∞", "▪", "●", "○", "◦", "*", "·"}
_ROMAN = {"i": 1, "v": 5, "x": 10}
INDENT_TOL = 4.0


def _roman(tok: str) -> int | None:
    """Value of a canonical lower-case roman numeral up to 39."""
    if not tok or any(c not in _ROMAN for c in tok):
        return None
    total = 0
    for a, b in zip(tok, tok[1:] + " "):
        v = _ROMAN[a]
        total += -v if b != " " and _ROMAN[b] > v else v
    canon = ("x" * (total // 10)
             + ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii",
                "ix"][total % 10])
    return total if 0 < total < 40 and canon == tok else None


@dataclass
class Enum:
    raw: str                       # as printed: "a)", "3.8.1", "(ii)"
    fmt: str                       # "dec" | "pdec" | "X." | "X)" | "(X)" | "bullet"
    token: str                     # "a", "3.8.1", "ii"
    interps: list[tuple[str, int]] = field(default_factory=list)
    dec: tuple[int, ...] | None = None
    path: tuple | None = None      # "(A.1)" -> ("A", 1), ASME sub-paragraphs
    titled: bool = False        # number followed by a title on its own line


_DEC = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){1,5})\.?$")
# "(A.1)", "(C.3.1)". Scanned codes often lose the dot or read 1 as l/I
# ("(Al)", "(Al.l)", "(A5)"), so both are tolerated.
_PDEC = re.compile(r"^\(([A-Z])\.?([\dlI]{1,2}(?:\.[\dlI]{1,2}){0,3})\)$")
# "104 PRESSURE DESIGN OF COMPONENTS": a bare section number is trusted
# only in front of an all-capitals title.
_SECTION = re.compile(r"^(\d{3})\s+([A-Z][A-Z0-9 ,/&()\-]{3,})$")
# "100.2 Definitions", "104.1.2 Straight Pipe Under Internal Pressure": a
# paragraph number followed by a short Title Case title is a heading.
_WORD = r"(?:[A-Z][\w\-]*|and|or|of|for|the|in|on|to|under|with|due)"
_TITLED = re.compile(rf"^(\d{{3}}(?:\.\d+)+)\s+({_WORD}(?:\s+{_WORD}){{0,7}})$")
# "arc welding: a group of welding processes ..." — a defined term
_TERM = re.compile(r"^[a-z][\w ,\-/()']{0,60}:\s+\S")
# "(A)Minimum Wall Thickness": OCR dropped the space after the enumerator
_GLUED = re.compile(r"^(\((?:[A-Z]|\d{1,2})(?:\.\d+)*\))([A-Z][a-z].*)$")
# a note, whose text no equation should be glued to
_NOTE = re.compile(r"^(?:GENERAL\s+)?NOTES?\b|^\[\d+\]", re.I)
_FORMS = (("X.", re.compile(r"^([A-Za-z]{1,5}|\d{1,3})\.$")),
          ("X)", re.compile(r"^([A-Za-z]{1,5}|\d{1,3})\)$")),
          ("(X)", re.compile(r"^\(([A-Za-z]{1,5}|\d{1,3})\)$")))
# a title names things; it does not oblige or define
_STATES = re.compile(r"\b(shall|should|must|may|will)\b|=", re.I)


def detect_enum(text: str) -> Enum | None:
    """Parse `text` as a complete enumerator, or return None."""
    t = text.strip()
    if t in BULLETS:
        return Enum(t, "bullet", t)
    m = _DEC.match(t)
    if m:
        return Enum(t, "dec", m.group(1),
                    dec=tuple(int(p) for p in m.group(1).split(".")))
    m = _PDEC.match(t)
    if m:
        nums = [int(n.replace("l", "1").replace("I", "1"))
                for n in m.group(2).split(".")]
        if len(nums) == 1 and nums[0] > 9 and "." not in t:
            nums = [nums[0] // 10, nums[0] % 10]      # "(A11)" is "(A.1.1)"
        return Enum(t, "pdec", ".".join([m.group(1), *map(str, nums)]),
                    path=(m.group(1), *nums))
    for fmt, rx in _FORMS:
        m = rx.match(t)
        if not m:
            continue
        tok = m.group(1)
        if tok.isdigit():
            return Enum(t, fmt, tok, [("num", int(tok))])
        interps: list[tuple[str, int]] = []
        case = "lower" if tok.islower() else "upper" if tok.isupper() else ""
        if not case:
            return None
        low = tok.lower()
        if len(tok) == 1:
            interps.append((f"alpha-{case}", ord(low) - 96))
        r = _roman(low)
        if r:
            interps.append((f"roman-{case}", r))
        return Enum(t, fmt, tok, interps) if interps else None
    return None


_INLINE = re.compile(r"^(\S{1,12})\s+(\S.*)$")


def split_enum(row: Row) -> tuple[Enum | None, str, float | None, bool]:
    """(enumerator, body text, body x, inline?) for a row.

    `inline` is True when the enumerator was cut out of a longer text run
    rather than being its own fragment — a weaker signal, so the tree only
    accepts it when it fits an open sequence."""
    parts = [p for p in row.parts if p.text.strip()]
    if not parts:
        return None, "", None, False
    e = detect_enum(parts[0].text)
    if e:
        body = " ".join(p.text for p in parts[1:])
        bx = parts[1].x0 if len(parts) > 1 else None
        return e, re.sub(r"\s+", " ", body).strip(), bx, False
    # ASME headings: the number shares the line with the title and the
    # paragraphs below start at the same margin, so the body x is the row x
    m = _SECTION.match(row.text)
    if m:
        e = Enum(m.group(1), "dec", m.group(1), dec=(int(m.group(1)),),
                 titled=True)
        return e, m.group(2).strip(), row.x0, False
    m = _TITLED.match(row.text)
    if m:
        e = Enum(m.group(1), "dec", m.group(1),
                 dec=tuple(int(p) for p in m.group(1).split(".")), titled=True)
        return e, m.group(2).strip(), row.x0, False
    m = _INLINE.match(row.text)
    if m:
        e = detect_enum(m.group(1))
        if e:
            return e, m.group(2).strip(), None, True
    m = _GLUED.match(row.text)
    if m:
        e = detect_enum(m.group(1))
        if e:
            return e, m.group(2).strip(), None, True
    return None, row.text, None, False


@dataclass
class Node:
    enum: Enum | None = None
    kind: str = ""                 # chosen interpretation: "alpha-lower", ...
    ord: int = 0
    x: float = 0.0                 # enumerator x, page-offset removed
    body_x: float = 0.0
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    is_heading: bool = False
    inline: bool = False           # enumerator shared a fragment with its text
    note: bool = False             # a footnote (from the layout model)
    # a non-clause element (table / figure / equation) placed in the flow
    payload: dict | None = None

    @property
    def text(self) -> str:
        out = ""
        for t in self.texts:
            if not t:
                continue
            if out.endswith("-") and t[:1].islower():
                out += t                      # "side-" + "step"
            else:
                out += (" " if out else "") + t
        return out

    def dec_key(self) -> tuple[int, ...] | None:
        """Decimal identity used to parent "3.1" under "3." or "3.8"."""
        if not self.enum:
            return None
        if self.enum.dec:
            return self.enum.dec
        if self.enum.fmt == "X." and self.kind == "num":
            return (self.ord,)
        return None

    def path_key(self) -> tuple | None:
        """Identity used to parent "(A.1)" under "(A)" or "(A.1.1)" under
        "(A.1)"."""
        if not self.enum:
            return None
        if self.enum.path:
            return self.enum.path
        if self.enum.fmt == "(X)" and self.kind == "alpha-upper":
            return (self.enum.token,)
        return None


class ClauseTree:
    def __init__(self, pitch: float, size: float):
        self.root = Node()
        self.stack: list[Node] = [self.root]
        self.cur: Node | None = None
        self.pitch = pitch             # typical distance between text rows
        self.size = size               # typical font size
        self._last: Row | None = None
        self._after_glue = False
        self._recent: list[Node] = []   # last clause nodes created, in order
        self._section: Node | None = None   # last titled heading seen
        # every decimal-numbered node by its number. Decimal numbering is
        # global, so "104.1.3" finds "104.1" even after un-numbered text
        # has closed the levels in between.
        self._dec_index: dict[tuple[int, ...], Node] = {}

    # ---- placement ---------------------------------------------------------
    def _attach(self, parent: Node, node: Node) -> Node:
        if any(n is parent for n in self.stack):
            while self.stack[-1] is not parent:
                self.stack.pop()
        else:                          # re-open a level that had been closed
            chain, n = [], parent
            while n is not None:
                chain.append(n)
                n = n.parent
            self.stack = chain[::-1]
        node.parent = parent
        parent.children.append(node)
        self.stack.append(node)
        self._recent = (self._recent + [node])[-40:]
        if node.enum is not None and node.enum.titled:
            self._section = node
        key = node.dec_key()
        if key and (node.enum.dec or parent is self.root):
            self._dec_index[key] = node
        return node

    def _place_path(self, e: Enum) -> Node:
        t = e.path
        assert t
        open_nodes = self.stack[1:]
        for n in reversed(open_nodes):                 # child by prefix
            key = n.path_key()
            if key and len(t) == len(key) + 1 and t[:len(key)] == key:
                return n
        for n in reversed(open_nodes):                 # sibling, same depth
            if n.enum and n.enum.path and n.parent is not None \
                    and len(n.enum.path) == len(t):
                return n.parent
        for n in reversed(open_nodes):                 # under a shorter prefix
            key = n.path_key()
            if key and len(t) > len(key) and t[:len(key)] == key:
                return n
        while len(self.stack) > 1 and (self.stack[-1].enum is None
                                       or self.stack[-1].path_key()):
            self.stack.pop()
        return self.stack[-1]

    def _place_decimal(self, e: Enum, mode: str) -> Node | None:
        """Parent for a decimal number, or None to treat it as plain text.

        `mode`: "free" for an enumerator that is its own fragment; "strict"
        for one cut out of running text, which must be the next number in
        its sequence (a wrapped line may start with "4.4 kN"); "plausible"
        for one followed by a capitalised title, which must at least move
        forward under a known parent."""
        t = e.dec
        assert t
        if len(t) == 1:                                # "104 GENERAL"
            return self.root
        parent = self._dec_index.get(t[:-1])
        if parent is not None:
            sibs = [c.enum.dec[-1] for c in parent.children
                    if c.enum and c.enum.dec]
            if mode == "strict" and not (
                    t[-1] == (sibs[-1] + 1 if sibs else 1)
                    or (not sibs and t[-1] == 0)):
                return None
            return parent
        for key in reversed(self._dec_index):          # sibling, same prefix
            if len(key) == len(t) and key[:-1] == t[:-1]:
                if mode != "free" and t[-1] <= key[-1]:
                    continue
                if mode == "strict" and t[-1] != key[-1] + 1:
                    continue
                return self._dec_index[key].parent
        if mode != "free" and self._dec_index:
            return None
        return self.root

    def _place_list(self, e: Enum, x: float, strict: bool, bold: bool
                    ) -> tuple[Node, str, int] | None:
        open_nodes = self.stack[1:]
        # (a) continues an open sequence
        for n in reversed(open_nodes):
            if n.enum and n.enum.fmt == e.fmt and n.parent is not None:
                for kind, o in e.interps:
                    if kind == n.kind and o == n.ord + 1:
                        return n.parent, kind, o
        # (b) opens a new level with its first member
        firsts = [(k, o) for k, o in e.interps if o == 1]
        if firsts:
            # a new level cannot sit left of the item it would nest under,
            # and a bold "1." after plain paragraphs is a heading, not a
            # list inside the last paragraph
            while len(self.stack) > 1 and (
                    self.stack[-1].x > x + INDENT_TOL
                    or (bold and self.stack[-1].enum is None)):
                self.stack.pop()
            return self.stack[-1], firsts[0][0], 1
        if strict:
            return None
        # (c) out of sequence (an extract that starts mid-list, a skipped
        # item): same format and kind anywhere above, else nest by indent
        for n in reversed(open_nodes):
            if n.enum and n.enum.fmt == e.fmt and n.parent is not None:
                for kind, o in e.interps:
                    if kind == n.kind and o > n.ord:
                        return n.parent, kind, o
        while len(self.stack) > 1 and self.stack[-1].x >= x - INDENT_TOL \
                and self.stack[-1].enum is not None:
            self.stack.pop()
        kind, o = e.interps[0]
        return self.stack[-1], kind, o

    # ---- feeding -----------------------------------------------------------
    def _in_definitions(self) -> bool:
        """Under a "100.2 Definitions" heading, where each term starts a
        paragraph that is set with no indent and no extra space."""
        s = self._section
        return s is not None and len(s.text) < 40 and "definition" in s.text.lower()

    @staticmethod
    def _is_note(n: Node) -> bool:
        while n is not None:
            if n.note or _NOTE.match(n.text or ""):
                return True
            n = n.parent
        return False

    def _is_continuation(self, row: Row, x: float, new_page: bool) -> bool:
        cur = self.cur
        if cur is None or cur.payload is not None or not cur.rows:
            return False
        if (row.source == "footnote") != cur.note:
            return False
        if _TERM.match(row.text) and self._in_definitions():
            return False
        if x < cur.body_x - INDENT_TOL:
            # a paragraph whose first line is indented: its second row sits
            # left of the first and still belongs to it
            gap = row.y0 - self._last.y0 if self._last else 0.0
            if (cur.enum is None or cur.inline) and len(cur.rows) == 1 \
                    and not new_page and 0 <= gap <= 1.3 * self.pitch:
                cur.body_x = x
                if cur.enum is None:
                    cur.x = x
                return True
            return False
        if new_page:
            return not cur.text.rstrip().endswith((".", ":", ";"))
        if self._after_glue:
            # the previous row was a tall equation: measure from its bottom
            return row.y0 - self._last.y1 <= 0.9 * self.pitch
        gap = row.y0 - self._last.y0 if self._last else 0.0
        limit = 1.3 * self.pitch * max(row.size, self.size) / self.size
        return 0 <= gap <= limit

    def add_row(self, row: Row, offset: float, glue: bool = False,
                col2: tuple[float, float] | None = None) -> None:
        """`offset` is the page's left shift (odd/even margins differ);
        `col2` = (split x, offset) for rows in a right-hand column. `glue`
        rows (display equations) always join the clause they sit in."""
        if is_toc_row(row.text):
            return
        if col2 is not None and row.x0 >= col2[0]:
            offset = col2[1]
        if glue and self.cur is not None and self.cur.payload is None:
            # the equation belongs to the last clause that is not a note:
            # a table and its notes may be printed between a clause and
            # the equation it introduces
            owner = self.cur
            for n in reversed(self._recent):
                if n.payload is None and not self._is_note(n):
                    owner = n
                    break
            owner.rows.append(row)
            owner.texts.append(row.text)
            self.cur = owner
            self._last, self._after_glue = row, True
            return
        new_page = self._last is not None and self._last.page != row.page
        x = row.x0 - offset
        if not new_page and self._last is not None \
                and row.y0 < self._last.y0 - 150 and row.x0 > self._last.x0 + 100:
            new_page = True            # top of the next column
        e, body, body_x, inline = split_enum(row)
        if body_x is not None:
            bx = body_x - offset
        elif inline and e and e.fmt == "bullet":
            bx = x + self.size         # text hangs to the right of the bullet
        elif inline:
            bx = x                     # no hanging indent: wraps to the margin
        else:
            bx = x + 2 * self.size

        def node_for(**kw) -> Node:
            return Node(enum=e, x=x, body_x=bx, rows=[row], texts=[body],
                        inline=inline, note=row.source == "footnote", **kw)

        if e and e.fmt == "bullet":
            parent = None
            for n in reversed(self.stack[1:]):
                if n.enum and n.enum.fmt == "bullet" and \
                        n.enum.token == e.token:
                    parent = n.parent
                    break
            if parent is None:
                while len(self.stack) > 1 and \
                        self.stack[-1].body_x > x + INDENT_TOL and \
                        self.stack[-1].enum is None:
                    self.stack.pop()
                parent = self.stack[-1]
            self.cur = self._attach(parent, node_for())
        elif e and e.fmt == "pdec":
            self.cur = self._attach(self._place_path(e), node_for())
        elif e and e.fmt == "dec":
            mode = "free" if not inline else (
                "plausible" if body[:1].isupper() else "strict")
            parent = self._place_decimal(e, mode)
            if parent is None:
                e = None
            else:
                self.cur = self._attach(parent, node_for())
        elif e:
            placed = self._place_list(e, x, strict=inline, bold=row.bold)
            if placed is None:
                e = None
            else:
                parent, kind, o = placed
                self.cur = self._attach(parent, node_for(kind=kind, ord=o))

        if e is None:
            if self._is_continuation(row, x, new_page):
                self.cur.rows.append(row)
                self.cur.texts.append(row.text)
            else:
                # un-numbered paragraph: child of the innermost open item
                # whose body starts at or left of this row
                while len(self.stack) > 1 and (
                        self.stack[-1].body_x > x + INDENT_TOL
                        or self.stack[-1].enum is None):
                    self.stack.pop()
                node = Node(x=x, body_x=x, rows=[row], texts=[row.text],
                            note=row.source == "footnote")
                self.cur = self._attach(self.stack[-1], node)
        self._last, self._after_glue = row, False

    def add_payload(self, payload: dict, top_level: bool = False) -> None:
        """Place a table or figure under the section it appears in (the
        nearest open heading), or at the document root for whole drawing
        sheets. The open list levels are left untouched so a list that
        continues after the figure still finds its siblings."""
        parent = self.root
        if not top_level:
            for n in reversed(self.stack[1:]):
                words = n.text.split()
                if n.enum and n.enum.fmt != "bullet" and 0 < len(words) <= 14 \
                        and not n.text.rstrip().endswith((".", ":", ";")):
                    parent = n
                    break
        node = Node(payload=payload, parent=parent)
        parent.children.append(node)
        self.cur = node

    # ---- finishing ---------------------------------------------------------
    def finish(self) -> Node:
        def mark(n: Node) -> None:
            for c in n.children:
                mark(c)
            if n.enum is None or n.enum.fmt == "bullet" or n.payload:
                return
            words = n.text.split()
            short = 0 < len(words) <= 14
            closed = n.text.rstrip().endswith((".", ":", ";"))
            bold = bool(n.rows) and all(r.bold for r in n.rows)
            states = bool(_STATES.search(n.text))
            n.is_heading = short and not closed and not states and (
                bool(n.children) or bold)
        mark(self.root)
        return self.root


def node_ref(chain: list[Node], page: int) -> str:
    """Citation for the last node of `chain` (root excluded)."""
    numbered = [n for n in chain if n.enum and n.enum.fmt != "bullet"]
    if numbered:
        last_dec = max((i for i, n in enumerate(numbered) if n.enum.dec),
                       default=None)
        if last_dec is None:
            head, rest = "", numbered
        else:
            head, rest = numbered[last_dec].enum.token, numbered[last_dec + 1:]
        tokens: list[str] = []
        for n in rest:
            tok = n.enum.token
            if n.enum.path:            # "A.1" supersedes its ancestor "A"
                tokens = [t for t in tokens
                          if not tok.startswith(t + ".")]
            tokens.append(tok)
        ref = " ".join(p for p in (head, ".".join(tokens)) if p)
    else:
        ref = f"p{page}"
    # un-numbered nodes below the last numbered one: "2.2 •1 •2", "1.2 ¶3"
    trailing: list[Node] = []
    for n in reversed(chain):
        if n.enum and n.enum.fmt != "bullet":
            break
        trailing.append(n)
    for n in reversed(trailing):
        sibs = [c for c in n.parent.children
                if (c.enum is None or c.enum.fmt == "bullet") and not c.payload]
        ref += f" {'•' if n.enum else '¶'}{sibs.index(n) + 1}"
    return ref
