"""Render the plain-language team document. Optional dependency: reportlab."""
from pathlib import Path
from html import escape
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor('#172638'); MUTE = colors.HexColor('#536578')
PALE = colors.HexColor('#EAF4F1'); LINE = colors.HexColor('#D7E1E8')

def inline(s):
    s = escape(s).replace('–','-').replace('—','-')
    s = re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<link href="\2" color="#167D70">\1</link>',s)
    s = re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',s)
    return re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',s)

def footer(c,doc):
    c.saveState();c.setStrokeColor(LINE);c.line(42,35,A4[0]-42,35)
    c.setFont('Helvetica',8);c.setFillColor(MUTE)
    c.drawString(42,22,'BarcelonAI | More than a Hack 2027 | 15 November 2026')
    c.drawRightString(A4[0]-42,22,str(doc.page));c.restoreState()

def main():
    st = {
        'title':ParagraphStyle('t',fontName='Helvetica-Bold',fontSize=24,leading=29,textColor=INK,spaceAfter=10),
        'h2':ParagraphStyle('h',fontName='Helvetica-Bold',fontSize=12.5,leading=16,textColor=INK,spaceBefore=11,spaceAfter=6,keepWithNext=True),
        'body':ParagraphStyle('b',fontName='Helvetica',fontSize=10,leading=14,textColor=INK,spaceAfter=8),
        'cell':ParagraphStyle('c',fontName='Helvetica',fontSize=9,leading=12,textColor=INK),
        'head':ParagraphStyle('ch',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=colors.white),
        'code':ParagraphStyle('code',fontName='Courier',fontSize=7.8,leading=10.5,textColor=INK,spaceAfter=2),
        'small':ParagraphStyle('s',fontName='Helvetica',fontSize=8.5,leading=11.5,textColor=MUTE,spaceAfter=7),
    }
    lines=(ROOT/'BarcelonAI.md').read_text().splitlines();flow=[];i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line in ('## Benchmarks and our results','## Testing on different matches','## Technical details and reproduction'):flow.append(PageBreak())
        if line.startswith('# '):
            flow.append(Paragraph(inline(line[2:]),st['title']))
            flow.append(Paragraph('The task, the results and how to reproduce them.',st['small']))
        elif line.startswith('## '):flow.append(Paragraph(inline(line[3:]),st['h2']))
        elif line.startswith('### '):flow.append(Paragraph(inline(line[4:]),st['h2']))
        elif line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
            cells=[]
            for row in rows:
                bits=[b.strip() for b in row.strip('|').split('|')]
                if all(re.fullmatch(r':?-+:?',b) for b in bits):continue
                style=st['head'] if not cells else st['cell']
                cells.append([Paragraph(inline(b),style) for b in bits])
            widths=[187,108,108,108] if len(cells[0])==4 else [245,133,133]
            t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('BACKGROUND',(0,1),(-1,-1),PALE),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
                ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,1),(-1,-1),.5,colors.white)]))
            flow.extend([t,Spacer(1,10)])
        elif line.startswith('```'):
            while i<len(lines) and not lines[i].startswith('```'):
                flow.append(Paragraph(escape(lines[i]),st['code']));i+=1
            i+=1;flow.append(Spacer(1,7))
        elif line.startswith('Expected CSV SHA-256:'):
            digest=re.search(r'`([^`]+)`',line).group(1)
            flow.append(Paragraph('Expected CSV SHA-256:',st['small']))
            flow.append(Paragraph(digest[:32]+'<br/>'+digest[32:],st['code']))
        else:flow.append(Paragraph(inline(line),st['body']))
    out=ROOT/'output/pdf/BarcelonAI.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=37,bottomMargin=48,
        title='BarcelonAI: method and results',author='BarcelonAI',subject='Football possession inference')
    doc.build(flow,onFirstPage=footer,onLaterPages=footer);print(out)

if __name__=='__main__':main()
