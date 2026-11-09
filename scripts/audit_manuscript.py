"""
Structural audit of the manuscript's accepted text: looks for the class of
defect that produced "about 59 years - a difference of roughly 5 years", i.e.
a sentence whose opening clause was deleted in an earlier round and whose tail
still reads grammatically on its own.

Checks, per paragraph of the accepted view:

  1. Truncated openings   - paragraph starts lowercase, or with a bare number
                            or connective, suggesting a removed lead-in.
  2. Run-together text    - "word.Word" with no space, the signature of a
                            deleted run leaving two sentences fused.
  3. Unterminated endings - paragraph ends without terminal punctuation.
  4. Repeated words       - "the the", introduced by partial replacements.
  5. Dangling references  - every Table / Figure / Section / Appendix mentioned
                            in the body resolves to something that exists.
  6. Citations            - every [n] used exists in the bibliography, and
                            every bibliography entry is cited somewhere.

Headings, captions and table cells are excluded from the prose checks, since
they legitimately lack terminal punctuation and often begin with a number.
"""
import re
import sys
import zipfile

W_T = re.compile(r'<w:t\b[^>]*>(.*?)</w:t>', re.DOTALL)
P_RE = re.compile(r'<w:p[ >].*?</w:p>', re.DOTALL)
DEL = re.compile(r'<w:del\b[^>]*>((?:(?!<w:ins\b|<w:del\b).)*?)</w:del>', re.DOTALL)
INS = re.compile(r'<w:ins\b[^>]*>((?:(?!<w:ins\b|<w:del\b).)*?)</w:ins>', re.DOTALL)


def accepted_xml(path):
    x = zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')
    cur = x
    while True:
        n = DEL.sub('', cur)
        n = INS.sub(r'\1', n)
        if n == cur:
            break
        cur = n
    return re.sub(r'<w:ins\b[^>]*/>', '', cur)


def paragraphs(xml):
    """Return (text, in_table). A paragraph is 'in a table' when its start
    offset falls inside a <w:tbl>...</w:tbl> span - the paragraph blob itself
    carries no marker, which is why a naive substring test classifies every
    table cell as prose."""
    spans = []
    depth = 0
    start = None
    for m in re.finditer(r'<w:tbl>|</w:tbl>', xml):
        if m.group(0) == '<w:tbl>':
            if depth == 0:
                start = m.start()
            depth += 1
        else:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, m.end()))
                start = None

    def in_table(pos):
        return any(a <= pos < b for a, b in spans)

    out = []
    for m in P_RE.finditer(xml):
        txt = ''.join(W_T.findall(m.group(0))).strip()
        if txt:
            out.append((txt, in_table(m.start())))
    return out


HEADING = re.compile(r'^(?:\d+(?:\.\d+)*\.?\s|Table\s|Figure\s|Appendix\s|Abstract|Keywords|References|'
                     r'Availability|Funding|Code Availability|Declarations|Acknowledg)', re.I)
# Only openers that cannot legitimately begin a sentence. "While", "Although",
# "However", "Therefore", "Because" are normal sentence starts and were removed
# after they produced only false positives.
CONNECTIVE = re.compile(r'^(and|but|or|which|whereas|about|approximately|versus|compared)\b', re.I)


def main(path):
    xml = accepted_xml(path)
    paras = paragraphs(xml)
    body = [t for t, intbl in paras if not intbl]
    full = '\n'.join(t for t, _ in paras)

    print(f'{len(paras)} paragraphs ({len(body)} outside tables)\n')

    print('--- 1. truncated openings (lowercase / bare number / connective start) ---')
    hits = 0
    for t in body:
        if HEADING.match(t):
            continue
        if re.fullmatch(r'\(?[a-c]\)?', t.strip()):
            continue  # figure sub-labels
        if re.match(r'^\[\d+\]', t):
            continue  # bibliography entries
        first = t.lstrip('“"\'([')
        if not first:
            continue
        if first[0].islower() or CONNECTIVE.match(first) or re.match(r'^\d+[^.)]', first):
            hits += 1
            print(f'   ! {t[:150]}')
    print('   none' if not hits else f'   {hits} flagged')

    print('\n--- 2. run-together sentences (no space after full stop) ---')
    hits = 0
    for t in body:
        for m in re.finditer(r'[a-z0-9]\.[A-Z][a-z]', t):
            s = max(0, m.start() - 60)
            hits += 1
            print(f'   ! ...{t[s:m.end()+60]}...')
    print('   none' if not hits else f'   {hits} flagged')

    print('\n--- 3. paragraphs not ending in terminal punctuation ---')
    hits = 0
    for t in body:
        if HEADING.match(t) or len(t) < 45:
            continue
        if not re.search(r'[.!?:”"\)\]]\s*$', t):
            hits += 1
            print(f'   ! ...{t[-120:]}')
    print('   none' if not hits else f'   {hits} flagged')

    print('\n--- 4. repeated words ---')
    rep = re.findall(r'\b(\w+)\s+\1\b', full)
    rep = [r for r in rep if r.lower() not in ('had', 'that', 'is')]
    print('   none' if not rep else f'   {rep}')

    print('\n--- 5. dangling cross-references ---')
    def targets(kind, pat):
        return set(re.findall(pat, full))
    tbl_defined = set(re.findall(r'Table ([A-Z]?\d+)\.', full))
    tbl_used = set(re.findall(r'Table ([A-Z]?\d+)\b', full))
    fig_defined = set(re.findall(r'Figure (\d+)\.', full))
    fig_used = set(re.findall(r'(?:Figure|Fig\.) (\d+)', full))
    sec_defined = set(re.findall(r'^(\d+(?:\.\d+)*)[. ]', '\n'.join(body), re.M))
    sec_used = set(re.findall(r'Section (\d+(?:\.\d+)*)', full))
    app_defined = set(re.findall(r'Appendix ([A-Z]):', full))
    app_used = set(re.findall(r'Appendix ([A-Z])\b', full))
    for lbl, used, defined in [('Table', tbl_used, tbl_defined), ('Figure', fig_used, fig_defined),
                               ('Section', sec_used, sec_defined), ('Appendix', app_used, app_defined)]:
        missing = sorted(used - defined)
        print(f'   {lbl:8s} used={len(used):<3} defined={len(defined):<3} unresolved={missing or "none"}')

    print('\n--- 6. citations ---')
    biblio = set(re.findall(r'^\[(\d+)\]', '\n'.join(body), re.M))
    cited = set(re.findall(r'\[(\d+)\]', full)) - biblio if False else set(re.findall(r'\[(\d+)\]', full))
    body_cited = set()
    for t in body:
        if re.match(r'^\[\d+\]', t):
            continue
        body_cited |= set(re.findall(r'\[(\d+)\]', t))
    print(f'   bibliography entries : {len(biblio)}')
    print(f'   cited in body        : {len(body_cited)}')
    print(f'   cited but not listed : {sorted(body_cited - biblio, key=lambda s: int(s)) or "none"}')
    print(f'   listed but not cited : {sorted(biblio - body_cited, key=lambda s: int(s)) or "none"}')


if __name__ == '__main__':
    main(sys.argv[1])
