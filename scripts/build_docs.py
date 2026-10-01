#!/usr/bin/env python3
"""Render the four tracked BeeOPS guides using only the Python standard library."""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import hashlib
import html
import os
import re
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
DOCUMENTS = (
    ('BeeOPS_조별기획서', 'BeeOPS 프로젝트 기획서', 'PROJECT PROPOSAL', '/proposal'),
    ('BeeOPS_모델_상세설명', 'BeeOPS 예측 모델 상세 설명', 'MODEL GUIDE', '/model-guide'),
    ('온도_드리프트_실습', '온도 변화와 실제 앙상블 재학습 실습', 'PRACTICE GUIDE', None),
    ('시연가이드', 'BeeOPS 최종본 6분 시연 가이드', 'DEMO GUIDE', None),
)
PDF_STEMS = tuple(stem for stem, _, _, route in DOCUMENTS if route)

def document_link(stem, label, route):
    label = html.escape(label)
    if route:
        return f'<a href="{route}" data-local-doc="{stem}.html">{label}</a>'
    # Supplementary documents are local artifacts, not registered HTTP routes.
    return f'<span class="source-ref" data-local-doc="{stem}.html" title="로컬 문서: {stem}.html">{label}</span>'


def inline(text):
    # Protect code spans before processing links and emphasis.
    protected = []
    def code(match):
        protected.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(protected)-1}\x00"
    text = re.sub(r"`([^`]+)`", code, text)
    text = html.escape(text)
    def link(match):
        label, target = match.groups()
        for stem, _, _, route in DOCUMENTS:
            if target == stem + '.md':
                return document_link(stem, html.unescape(label), route)
        if target.startswith(('https://', 'http://', '#')):
            return f'<a href="{target}" target="_blank" rel="noopener">{label}</a>'
        # Implementation evidence remains in the project. Avoid broken HTTP
        # links or making model files and runtime data publicly downloadable.
        return f'<span class="source-ref" title="프로젝트 근거 파일: {target}">{label}</span>'
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    for index, value in enumerate(protected):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def image_block(alt, relative):
    path = (DOCS / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Image outside BeeOPS: {relative}")
    if not path.exists():
        return f'<p class="missing">이미지 파일 대기: {html.escape(relative)}</p>'
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    payload = path.read_bytes()
    data = base64.b64encode(payload).decode("ascii")
    if mime == "image/png":
        width, height = struct.unpack('>II', payload[16:24])
    else:
        svg = ET.fromstring(payload)
        width, height = svg.attrib['width'], svg.attrib['height']
    return f'<figure><img src="data:{mime};base64,{data}" width="{width}" height="{height}" style="height:auto" alt="{html.escape(alt)}" loading="lazy"><figcaption>{html.escape(alt)}</figcaption></figure>'


def render_markdown(document):
    """Small deterministic renderer for the checked-in proposal's Markdown.

    Supported blocks: headings, paragraphs, lists, GFM-style tables, fenced code,
    blockquotes, images and exact details wrappers. No arbitrary HTML execution.
    """
    lines = document.splitlines()
    output, toc = [], []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("<!-- GENERATED:"):
            i += 1
            continue
        if line.startswith("```"):
            language = line[3:].strip()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            output.append(f'<pre class="code"><span class="code-language">{html.escape(language or "text")}</span><code>{html.escape(chr(10).join(block))}</code></pre>')
            i += 1
            continue
        heading = re.match(r"^(#{1,6}) (.+)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            anchor = f"section-{len(toc)+1}" if level == 2 else None
            if level == 2:
                toc.append((anchor, title))
            id_attribute = f' id="{anchor}"' if anchor else ""
            output.append(f"<h{level}{id_attribute}>{inline(title)}</h{level}>")
            i += 1
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            output.append(image_block(*image.groups()))
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"\|[\s:|\-]+\|$", lines[i+1].strip()):
            headers = [part.strip() for part in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([part.strip() for part in lines[i].strip().strip("|").split("|")])
                i += 1
            output.append('<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{inline(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>"+"".join(f"<td>{inline(c)}</td>" for c in row)+"</tr>" for row in rows) + "</tbody></table></div>")
            continue
        if line.startswith("> "):
            block = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                block.append(lines[i].strip()[2:])
                i += 1
            output.append("<blockquote>"+inline(" ".join(block))+"</blockquote>")
            continue
        if re.match(r"^(- |\d+\. )", line):
            ordered = bool(re.match(r"^\d+\. ", line))
            tag = "ol" if ordered else "ul"
            items = []
            pattern = r"^\d+\. " if ordered else r"^- "
            while i < len(lines) and re.match(pattern, lines[i].strip()):
                items.append(re.sub(pattern, "", lines[i].strip()))
                i += 1
            output.append(f"<{tag}>"+"".join(f"<li>{inline(item)}</li>" for item in items)+f"</{tag}>")
            continue
        details = re.fullmatch(r"<details><summary>([^<]+)</summary>", line)
        if details:
            output.append(f"<details><summary>{html.escape(details.group(1))}</summary>")
            i += 1
            continue
        if line == "</details>":
            output.append(line)
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|!\[|> |- |\d+\. |<)", lines[i].strip()):
            paragraph.append(lines[i].strip())
            i += 1
        output.append("<p>"+inline(" ".join(paragraph))+"</p>")
    return "\n".join(output), toc



def build():
    css = (DOCS / 'assets/document.css').read_text(encoding='utf-8')
    for stem, title, kind, route in DOCUMENTS:
        source = DOCS / (stem + '.md')
        document = source.read_text(encoding='utf-8')
        if '<!-- GENERATED:' in document:
            raise ValueError('Current documents must not contain historical generated blocks')
        body, toc = render_markdown(document)
        checksum = hashlib.sha256(document.encode()).hexdigest()
        epoch = os.environ.get('SOURCE_DATE_EPOCH')
        generated_at = datetime.fromtimestamp(int(epoch), timezone.utc) if epoch else datetime.now(timezone.utc)
        generated = generated_at.isoformat(timespec='seconds')
        navigation = ''.join(f'<a href="#{anchor}">{html.escape(label)}</a>' for anchor, label in toc)
        related = ''.join(document_link(other, label, link) for other, label, _, link in DOCUMENTS if other != stem)
        pdf_link = f'<a href="{route}.pdf" data-local-pdf="../output/pdf/{stem}.pdf">PDF 다운로드</a>' if route else ''
        page = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="source-sha256" content="{checksum}"><meta name="generated-at" content="{generated}"><title>{title}</title><style>{css}</style></head>
<body><aside><a class="brand" href="/">BeeOPS</a><div class="nav-kicker">{kind}</div><nav aria-label="문서 목차">{navigation}</nav><div class="related">{related}{pdf_link}<button type="button" onclick="window.print()">인쇄 또는 PDF 저장</button></div></aside><div class="wrap"><main><div class="topline"><span>SKALA · MODEL SERVING AND AIOPS</span><span>2026. 10. 01.</span></div>{body}<footer class="footer"><p>BeeOPS · {title} · 구현 근거 파일의 경로와 링크는 같은 이름의 Markdown 원문에서 확인할 수 있습니다.</p><p>문서 생성 {generated} · 원문 SHA-256 {checksum}</p></footer></main></div><script>if(location.protocol==='file:'){{document.querySelectorAll('[data-local-doc]').forEach(el=>{{const a=document.createElement('a');a.href=el.dataset.localDoc;a.textContent=el.textContent;el.replaceWith(a);}});document.querySelector('.brand').href='BeeOPS_조별기획서.html';document.querySelectorAll('a[data-local-pdf]').forEach(a=>a.href=a.dataset.localPdf);}}</script></body></html>'''
        destination = DOCS / (stem + '.html')
        destination.write_text(page, encoding='utf-8')
        print(f'Built {destination.relative_to(ROOT)}: {len(toc)} sections, {page.count("data:image/")} figures')

if __name__ == '__main__':
    build()
