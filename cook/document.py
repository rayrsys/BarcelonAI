"""Render the team report (BarcelonAI.md -> output/pdf/BarcelonAI.pdf). Dependency: reportlab only.

    python -m cook.document            # from the repository root

The text comes from BarcelonAI.md; this file adds the layout: a cover band, headline figures, a
pipeline diagram, a score chart and a data-split diagram. [[pipeline]], [[split]] and [[chart ...]]
lines in the Markdown place the figures; the chart reads its values from the table above it.
"""
import re
import sys
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, KeepTogether, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle, PageBreak)

ROOT = Path(__file__).resolve().parents[1]
W, H = A4
M = 46                      # page margin
CW = W - 2 * M              # content width
INK = colors.HexColor("#14213D"); TEXT = colors.HexColor("#243447"); MUTE = colors.HexColor("#5B6B7C")
TEAL = colors.HexColor("#167D70"); PALE = colors.HexColor("#E8F3F0"); LINE = colors.HexColor("#D5DEE6")
AMBER = colors.HexColor("#E0A030"); GREY = colors.HexColor("#A9B4BF"); SOFT = colors.HexColor("#F4F6F8")

S = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.3, leading=12.8, textColor=TEXT, spaceAfter=6.4, allowWidows=0, allowOrphans=0),
    "lead": ParagraphStyle("lead", fontName="Helvetica", fontSize=9.3, leading=12.8, textColor=TEXT, spaceAfter=7, keepWithNext=True),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.3, leading=12.8, textColor=TEXT, spaceAfter=3, leftIndent=14, bulletIndent=3, allowWidows=0, allowOrphans=0),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=INK, spaceAfter=7),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.8, leading=14, textColor=TEAL, spaceBefore=6, spaceAfter=5, keepWithNext=True),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=12, textColor=TEXT),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=INK),
    "num": ParagraphStyle("num", fontName="Helvetica", fontSize=9, leading=12, textColor=TEXT, alignment=2),
    "numb": ParagraphStyle("numb", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=INK, alignment=2),
    "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=7.6, leading=10, textColor=MUTE),
    "thn": ParagraphStyle("thn", fontName="Helvetica-Bold", fontSize=7.6, leading=10, textColor=MUTE, alignment=2),
    "cap": ParagraphStyle("cap", fontName="Helvetica", fontSize=8, leading=11, textColor=MUTE, spaceAfter=8),
    "kpi": ParagraphStyle("kpi", fontName="Helvetica-Bold", fontSize=21, leading=24, textColor=INK),
    "kpil": ParagraphStyle("kpil", fontName="Helvetica", fontSize=8, leading=10.5, textColor=MUTE),
    "call": ParagraphStyle("call", fontName="Helvetica", fontSize=9.3, leading=13.2, textColor=TEXT),
    "box": ParagraphStyle("box", fontName="Helvetica", fontSize=7.8, leading=9.8, textColor=TEXT, alignment=1),
    "boxb": ParagraphStyle("boxb", fontName="Helvetica-Bold", fontSize=8.2, leading=10, textColor=INK, alignment=1),
}


def inline(s):
    s = escape(s).replace("–", "-").replace("—", "-")
    s = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2" color="#167D70">\1</link>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    return re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.4">\1</font>', s)


class Cover(Flowable):
    def __init__(self, title, subtitle, team):
        super().__init__(); self.title, self.subtitle, self.team = title, subtitle, team; self.height = 100
    def wrap(self, aw, ah):
        self.width = aw; return aw, self.height
    def draw(self):
        c = self.canv
        c.setFillColor(INK); c.roundRect(0, 0, self.width, self.height, 8, stroke=0, fill=1)
        c.setFillColor(TEAL); c.rect(0, 0, 6, self.height, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#9FD8CE")); c.setFont("Helvetica-Bold", 8)
        c.drawString(24, self.height - 22, "MORE THAN A HACK 2027  ·  PHASE 1: BALL POSSESSION")
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 30); c.drawString(24, self.height - 54, self.title)
        c.setFont("Helvetica", 11); c.setFillColor(colors.HexColor("#DCE6EE")); c.drawString(24, self.height - 72, self.subtitle)
        c.setFont("Helvetica", 8.2); c.setFillColor(colors.HexColor("#B7C6D3")); c.drawString(24, 11, self.team)


class Rule(Flowable):
    """Short accent bar above a section heading."""
    def wrap(self, aw, ah):
        return aw, 12
    def draw(self):
        self.canv.setFillColor(TEAL); self.canv.rect(0, 3, 26, 2.6, stroke=0, fill=1)


class Bars(Flowable):
    """Horizontal bar chart. rows: (label, value, kind) with kind in base / ours / best; or grouped pairs."""
    def __init__(self, rows, vmax, title, pair_labels=None):
        super().__init__(); self.rows, self.vmax, self.title, self.pairs = rows, vmax, title, pair_labels
        self.bar = 9 if pair_labels else 12; self.gap = 7
        per = (2 * self.bar + 2) if pair_labels else self.bar
        self.height = 22 + len(rows) * (per + self.gap) + (14 if pair_labels else 4)
    def wrap(self, aw, ah):
        self.width = aw; return aw, self.height
    def draw(self):
        c = self.canv; lab_w = 130; x0 = lab_w; span = self.width - lab_w - 44
        c.setFont("Helvetica-Bold", 8.4); c.setFillColor(INK); c.drawString(0, self.height - 10, self.title)
        fill = {"base": GREY, "ours": colors.HexColor("#6FB5AA"), "best": TEAL}
        y = self.height - 22
        for row in self.rows:
            label, vals, kind = row[0], (row[1] if isinstance(row[1], (list, tuple)) else [row[1]]), row[2]
            per = len(vals) * self.bar + (len(vals) - 1) * 2
            c.setFont("Helvetica-Bold" if kind == "best" else "Helvetica", 8.4); c.setFillColor(INK if kind == "best" else TEXT)
            c.drawRightString(lab_w - 8, y - per / 2 - 3, label)
            for k, v in enumerate(vals):
                yy = y - (k + 1) * self.bar - k * 2
                c.setFillColor(SOFT); c.rect(x0, yy, span, self.bar, stroke=0, fill=1)
                col = fill[kind] if k == 0 or not self.pairs else colors.Color(fill[kind].red, fill[kind].green, fill[kind].blue, alpha=0.6)
                c.setFillColor(col); c.rect(x0, yy, span * v / self.vmax, self.bar, stroke=0, fill=1)
                c.setFillColor(INK if kind == "best" else TEXT); c.setFont("Helvetica-Bold" if kind == "best" else "Helvetica", 8)
                c.drawString(x0 + span * v / self.vmax + 5, yy + (self.bar - 6) / 2, f"{v:.3f}")
            y -= per + self.gap
        if self.pairs:
            c.setFont("Helvetica", 7.4); c.setFillColor(MUTE)
            c.drawString(x0, 2, f"Upper bar: {self.pairs[0]}.  Lower, lighter bar: {self.pairs[1]}.")


class Pipeline(Flowable):
    def __init__(self, steps):
        super().__init__(); self.steps = steps; self.height = 74
    def wrap(self, aw, ah):
        self.width = aw; return aw, self.height
    def draw(self):
        c = self.canv; n = len(self.steps); gap = 14; bw = (self.width - gap * (n - 1)) / n
        for i, (head, text) in enumerate(self.steps):
            x = i * (bw + gap); last = i == n - 1
            c.setFillColor(PALE if not last else INK); c.setStrokeColor(LINE if not last else INK)
            c.roundRect(x, 0, bw, self.height, 5, stroke=1, fill=1)
            hs = ParagraphStyle("h", parent=S["boxb"], textColor=colors.white if last else INK)
            ts = ParagraphStyle("t", parent=S["box"], textColor=colors.HexColor("#DCE6EE") if last else TEXT)
            ph = Paragraph(head, hs); _, hh = ph.wrap(bw - 10, 40); ph.drawOn(c, x + 5, self.height - 8 - hh)
            pt = Paragraph(text, ts); _, th = pt.wrap(bw - 10, 60); pt.drawOn(c, x + 5, self.height - 13 - hh - th)
            if not last:
                ax = x + bw + 3; ay = self.height / 2
                c.setStrokeColor(MUTE); c.setFillColor(MUTE); c.setLineWidth(1); c.line(ax, ay, ax + gap - 7, ay)
                p = c.beginPath(); p.moveTo(ax + gap - 3, ay); p.lineTo(ax + gap - 8, ay + 3); p.lineTo(ax + gap - 8, ay - 3); p.close()
                c.drawPath(p, stroke=0, fill=1)


class Split(Flowable):
    """Seven external matches: four to train, one to choose settings, two held out."""
    def __init__(self, groups):
        super().__init__(); self.groups = groups; self.height = 62
    def wrap(self, aw, ah):
        self.width = aw; return aw, self.height
    def draw(self):
        c = self.canv; n = sum(len(g[1]) for g in self.groups); gap = 6; ggap = 18
        bw = (self.width - gap * (n - len(self.groups)) - ggap * (len(self.groups) - 1)) / n; x = 0
        for name, ids, col, note in self.groups:
            gx = x
            for mid in ids:
                c.setFillColor(col); c.roundRect(x, 20, bw, 24, 4, stroke=0, fill=1)
                c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 8); c.drawCentredString(x + bw / 2, 29, mid)
                x += bw + gap
            x += ggap - gap
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 8.2); c.drawString(gx, 50, name)
            c.setFillColor(MUTE); c.setFont("Helvetica", 7.6); c.drawString(gx, 8, note)


class Heading(Flowable):
    """Section heading with its accent bar, as one flowable that stays with what follows."""
    keepWithNext = True
    def __init__(self, text):
        super().__init__(); self.p = Paragraph(inline(text), S["h2"])
    def wrap(self, aw, ah):
        w, h = self.p.wrap(aw, ah); self.ph = h; self.width = aw; self.height = h + 22; return aw, self.height
    def draw(self):
        self.canv.setFillColor(TEAL); self.canv.rect(0, self.ph + 10, 26, 2.6, stroke=0, fill=1)
        self.p.drawOn(self.canv, 0, 7)


def heading(text):
    return Heading(text)


def callout(html, colour=TEAL, fill=PALE):
    t = Table([[Paragraph(html, S["call"])]], colWidths=[CW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill), ("LINEBEFORE", (0, 0), (0, -1), 3, colour),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                           ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    return [t, Spacer(1, 9)]


def kpis(items):
    cells = [[[Paragraph(v, S["kpi"]), Spacer(1, 3), Paragraph(l, S["kpil"])] for v, l in items]]
    t = Table(cells, colWidths=[CW / len(items)] * len(items))
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
          ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("BOX", (0, 0), (-1, -1), 0.6, LINE)]
    st += [("LINEBEFORE", (i, 0), (i, 0), 0.6, LINE) for i in range(1, len(items))]
    t.setStyle(TableStyle(st)); return [t, Spacer(1, 6)]


def num(s):
    return float(re.sub(r"[^0-9.]", "", s))


def kind_of(label):
    low = label.lower()
    return "best" if low.startswith("method 2") else "ours" if low.startswith("method 1") else "base"


def decorate(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setFillColor(INK); c.rect(0, H - 9, W, 9, stroke=0, fill=1); c.setFillColor(TEAL); c.rect(0, H - 9, 90, 9, stroke=0, fill=1)
    c.setStrokeColor(LINE); c.setLineWidth(0.6); c.line(M, 36, W - M, 36)
    c.setFont("Helvetica", 7.8); c.setFillColor(MUTE)
    c.drawString(M, 23, "BarcelonAI  ·  More than a Hack 2027  ·  Technical report")
    c.drawRightString(W - M, 23, f"Page {doc.page}"); c.restoreState()


def is_num(cell):
    return bool(re.match(r"^[0-9]", cell.strip())) or cell.strip() in ("", "n/a")


def table(rows):
    """rows: raw strings, header first. Columns whose body cells are all numeric are right-aligned."""
    ncol = len(rows[0]); numeric = [k > 0 and all(is_num(r[k]) for r in rows[1:]) for k in range(ncol)]
    body = []
    for r, bits in enumerate(rows):
        strong = r > 0 and bits[0].lower().startswith("method 2")
        out = []
        for k, b in enumerate(bits):
            if r == 0:
                out.append(Paragraph(inline(b).upper(), S["thn"] if numeric[k] else S["th"]))
            elif numeric[k]:
                out.append(Paragraph(inline(b), S["numb"] if strong else S["num"]))
            else:
                out.append(Paragraph(inline(b), S["cellb"] if strong and k == 0 else S["cell"]))
        body.append(out)
    if ncol == 7:
        widths = [109, 49, 81, 72, 63, 57, 48]
    elif not any(numeric):
        widths = [CW * 0.27, CW * 0.2, CW * 0.53] if ncol == 3 else [CW / ncol] * ncol
    else:
        first = CW * (0.44 if ncol <= 3 else 0.36); widths = [first] + [(CW - first) / (ncol - 1)] * (ncol - 1)
    scale = CW / sum(widths); widths = [w * scale for w in widths]
    t = Table(body, colWidths=widths, hAlign="LEFT", repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LINEBELOW", (0, 0), (-1, 0), 1, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE)]
    for r, bits in enumerate(rows):
        if r > 0 and bits[0].lower().startswith("method 2"):
            st.append(("BACKGROUND", (0, r), (-1, r), PALE))
    t.setStyle(TableStyle(st)); return [t, Spacer(1, 9)]


def build(src, out):
    lines = src.read_text().splitlines(); flow = []; i = 0; last_table = None
    team = next((re.sub(r"\*\*Team:\*\*\s*", "", l).strip() for l in lines if l.startswith("**Team:**")), "")
    while i < len(lines):
        line = lines[i].rstrip(); i += 1
        if not line.strip() or line.startswith("**Team:**"):
            continue
        if line.startswith("# "):
            flow += [Cover(line[2:].strip(), "Two methods, the evidence and the final submission", team.rstrip(".")),
                     Spacer(1, 10)]
        elif line == "[[page]]":
            flow.append(PageBreak())
        elif line == "[[kpis external]]":
            flow.append(Paragraph("METHOD 2 · EQUAL-WEIGHT AVERAGES ACROSS TWO EXTERNAL TEST MATCHES", S["cap"]))
            flow += kpis([("0.594", "spellF1: possession timing and coverage (smooth50: 0.160)"),
                          ("87.6%", "of live frames assigned to the correct team"),
                          ("0.970", "live/dead F1: distinguishing active play from stoppages")])
        elif line.startswith("## "):
            title = line[3:].strip()
            if title == "Abstract":
                para = []
                while i < len(lines) and not lines[i].startswith("#"):
                    if lines[i].strip():
                        para.append(lines[i].strip())
                    i += 1
                flow += callout("<b>Abstract.</b> " + inline(" ".join(para)))
                continue
            flow.append(CondPageBreak(90)); flow.append(heading(title))
        elif line.startswith("### "):
            flow.append(CondPageBreak(70)); flow.append(Paragraph(inline(line[4:].strip()), S["h3"]))
        elif line.startswith("[[pipeline"):
            flow += [Pipeline([("Tactical tracking", "players and ball at 25 Hz; ball missing on 40.7% of graded frames"),
                               ("120 features", "ball proximity, credible control before and after, team shape, gaps, context"),
                               ("Boosted classifier", "trained on four external matches with simulated tracking failures"),
                               ("Viterbi decoding", "learned transition probabilities, one half at a time"),
                               ("H / A / D", "one label for each of 199,319 frames")]), Spacer(1, 10)]
        elif line.startswith("[[split"):
            flow += [Split([("Train", ["J03WPY", "J03WN1", "J03WMX", "J03WOH"], INK, "fit the classifier and transition probabilities"),
                            ("Development", ["J03WQQ"], AMBER, "selects the settings"),
                            ("Held-out test", ["J03WOY", "J03WR9"], TEAL, "after settings are fixed")]), Spacer(1, 8)]
        elif line.startswith("[[chart"):
            groups = {}
            for r in last_table[1:]:
                groups.setdefault(r[0], []).append(float(re.match(r"[0-9.]+", r[2]).group()))
            flow.append(KeepTogether([Bars([(k, v, kind_of(k)) for k, v in groups.items()], 0.7,
                                           "spellF1 on the two held-out matches", ("J03WOY", "J03WR9")), Spacer(1, 8)]))
        elif line.startswith("|"):
            raw = [line]
            while i < len(lines) and lines[i].startswith("|"):
                raw.append(lines[i]); i += 1
            rows = [[b.strip() for b in r.strip("|").split("|")] for r in raw]
            rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", b) for b in r)]
            last_table = rows; flow.append(KeepTogether(table(rows)))
        elif line.startswith("- "):
            flow.append(Paragraph(inline(line[2:]), S["bullet"], bulletText="\u2022"))
        elif line.startswith("```"):
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            pre = Preformatted("\n".join(code), ParagraphStyle("code", fontName="Courier", fontSize=7.6, leading=10.6, textColor=colors.HexColor("#E6EDF3")))
            t = Table([[pre]], colWidths=[CW])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), INK), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                   ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
            flow += [t, Spacer(1, 9)]
        elif line.startswith("Expected CSV SHA-256"):
            sha = re.search(r"`([0-9a-f]{64})`", line)
            flow.append(Paragraph("EXPECTED CSV SHA-256", S["th"]))
            flow.append(Paragraph(f'<font name="Courier" size="7.8">{sha.group(1) if sha else ""}</font>', S["body"]))
        elif line.startswith("Prepared for"):
            flow.append(Paragraph(inline(line), S["cap"]))
        else:
            flow.append(Paragraph(inline(line), S["lead"] if line.rstrip().endswith(":") else S["body"]))
    merged = []                       # keep a heading on the same page as the table or figure under it
    for fl in flow:
        prev = merged[-1] if merged else None
        is_head = isinstance(prev, Heading) or (isinstance(prev, Paragraph) and prev.style.name == "h3")
        if is_head and isinstance(fl, (KeepTogether, Pipeline, Split)):
            merged[-1] = KeepTogether([prev] + (list(fl._content) if isinstance(fl, KeepTogether) else [fl]))
        else:
            merged.append(fl)
    flow = merged
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=40, bottomMargin=50,
                          title="BarcelonAI: technical report", author="BarcelonAI")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(M, 50, CW, H - 90, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
                                       onPage=decorate)])
    doc.build(flow); return out


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "BarcelonAI.md"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output" / "pdf" / "BarcelonAI.pdf"
    print(build(src, out))


if __name__ == "__main__":
    main()
