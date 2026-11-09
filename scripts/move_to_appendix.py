"""
Relocate the dataset-cleansing logic and its three tables out of Section 3.1.1
into a new "Appendix A. Dataset Cleansing and Cohort Derivation" at the end of
the document, then renumber all tables to follow the new order of appearance.

Moved (a single contiguous span, verified to consist entirely of not-yet-accepted
<w:ins> content, so relocating it produces no tracked-change artefact — the text
simply appears as an insertion in its new place):

    * the cleansing paragraph (zero-coded missing values, cholesterol/BP
      exclusions, duplicate removal, IQR outliers, the 80/20 split)
    * Table (cohort derivation funnel) + caption
    * Table (per-variable missing / invalid values) + caption
    * the baseline-covariate discussion paragraph
    * Table (baseline covariate distributions) + caption

Left in Section 3.1.1: the harmonization/provenance paragraph, which is dataset
description rather than cleansing, plus a new forward pointer to Appendix A.

Renumbering after the move (order of first appearance):

    old 1  dataset constituents          -> 1   (unchanged, Section 3.1)
    old 5  XGBoost metrics               -> 2   (Section 3.3)
    old 6  downstream impact             -> 3   (Section 4.1)
    old 7  improvement-focused recourse  -> 4   (Section 4.10)
    old 8  ablation headline             -> 5   (Section 4.11)
    old 9  per-feature ablation          -> 6   (Section 4.11)
    old 10 stability vs inferential      -> 7   (Section 4.12)
    old 2  cohort derivation             -> 8   (Appendix A)
    old 3  per-variable missing/invalid  -> 9   (Appendix A)
    old 4  baseline covariates           -> 10  (Appendix A)

Appendix tables continue the single numeric sequence rather than taking an
"A1/A2/A3" style prefix, consistent with the instruction to keep one continuous
table sequence.
"""
import collections
import re
import sys

AUTHOR = "Prasad Kulkarni"
DATE = "2026-08-08T00:00:00Z"

PATH = 'v9unpacked/word/document.xml'

MAP = {
    'Table 1': 'Table 1',
    'Table 5': 'Table 2',
    'Table 6': 'Table 3',
    'Table 7': 'Table 4',
    'Table 8': 'Table 5',
    'Table 9': 'Table 6',
    'Table 10': 'Table 7',
    'Table 2': 'Table 8',
    'Table 3': 'Table 9',
    'Table 4': 'Table 10',
}

TOK = re.compile(r'Table\s+\d+')
T_NODE = re.compile(r'(<w:t\b[^>]*>)(.*?)(</w:t>)', re.DOTALL)
PARA = re.compile(r'<w:p[ >].*?</w:p>', re.DOTALL)

_id = [61000]


def nid():
    _id[0] += 1
    return _id[0]


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


RPR_BODY = ('<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
            'w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
            '<w:color w:val="111111"/><w:sz w:val="24"/><w:szCs w:val="24"/>')
RPR_HEAD = ('<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
            'w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b/>'
            '<w:color w:val="212529"/><w:sz w:val="24"/><w:szCs w:val="24"/>')


def ins_para(text, rpr=RPR_BODY, heading=False):
    ppr = ('<w:pPr>'
           + ('<w:pStyle w:val="Heading2"/>' if heading else '')
           + '<w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/>'
           + ('<w:spacing w:before="240" w:after="120" w:line="300" w:lineRule="auto"/>'
              if heading else
              '<w:spacing w:after="120" w:line="360" w:lineRule="auto"/>')
           + '<w:jc w:val="both"/>'
           + f'<w:rPr><w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}"/>{rpr}</w:rPr>'
             '</w:pPr>')
    return (f'<w:p>{ppr}'
            f'<w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">'
            f'<w:r><w:rPr>{rpr}</w:rPr>'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:ins></w:p>')


POINTER = ("The record-level cleansing applied to this distributed file, the resulting cohort "
           "derivation, and the baseline characteristics of the analysis cohort are reported in "
           "full in Appendix A (Tables 8, 9 and 10); the steps are summarised in Section 3.2 and "
           "the resulting cohort is used throughout Section 4.")

APPENDIX_HEAD = "Appendix A. Dataset Cleansing and Cohort Derivation"

APPENDIX_INTRO = ("This appendix documents the record-level processing applied to the distributed "
                  "dataset described in Section 3.1.1, the derivation of the 48-patient analysis "
                  "cohort, and the baseline characteristics of that cohort relative to the wider "
                  "dataset. It is placed here rather than in the main text because it concerns data "
                  "preparation rather than the recourse framework itself; no result reported in "
                  "Section 4 depends on any choice not documented in this appendix.")


def remap(text):
    return TOK.sub(lambda m: MAP.get(re.sub(r'\s+', ' ', m.group(0)), m.group(0)), text)


def assert_no_split_labels(xml):
    bad = 0
    for m in PARA.finditer(xml):
        nodes = [mm.group(2) for mm in T_NODE.finditer(m.group(0))]
        if nodes and len(TOK.findall(''.join(nodes))) > sum(len(TOK.findall(n)) for n in nodes):
            bad += 1
    if bad:
        raise SystemExit(f'ABORT: {bad} paragraph(s) have a table label split across runs.')


def main():
    xml = open(PATH, encoding='utf-8').read()

    # ---- 1. locate the block to move -------------------------------------
    a = xml.find('Missing and physiologically implausible values')
    b = xml.find('3.2 Data Exploration')
    if a < 0 or b < 0:
        raise SystemExit('ABORT: could not locate the cleansing block boundaries.')
    start = max(xml.rfind('<w:p ', 0, a), xml.rfind('<w:p>', 0, a))
    end = max(xml.rfind('<w:p ', 0, b), xml.rfind('<w:p>', 0, b))
    block = xml[start:end]

    n_tbl = block.count('<w:tbl>')
    if n_tbl != 3:
        raise SystemExit(f'ABORT: expected 3 tables in the moved block, found {n_tbl}.')
    baseline_runs = 0
    for m in re.finditer(r'<w:r>(.*?)</w:r>', block, re.DOTALL):
        if not re.search(r'<w:t\b|<w:delText\b', m.group(1)):
            continue
        depth = 0
        for t in re.finditer(r'<w:ins\b[^>]*>|</w:ins>', block[:m.start()]):
            depth += 1 if t.group(0).startswith('<w:ins') else -1
        if depth <= 0:
            baseline_runs += 1
    if baseline_runs:
        raise SystemExit(f'ABORT: {baseline_runs} baseline run(s) in block; move would need tracking.')

    print(f'block: {len(block)} chars, {n_tbl} tables, all content is unaccepted insertions')

    # ---- 2. remove from origin, add forward pointer -----------------------
    xml = xml[:start] + ins_para(POINTER) + xml[end:]

    # ---- 3. append Appendix A at end of body ------------------------------
    sect = xml.rfind('<w:sectPr')
    if sect < 0:
        raise SystemExit('ABORT: no <w:sectPr> found.')
    appendix = (ins_para(APPENDIX_HEAD, rpr=RPR_HEAD, heading=True)
                + ins_para(APPENDIX_INTRO)
                + block)
    xml = xml[:sect] + appendix + xml[sect:]

    # ---- 4. renumber ------------------------------------------------------
    assert_no_split_labels(xml)
    before = collections.Counter(TOK.findall(''.join(m.group(2) for m in T_NODE.finditer(xml))))
    xml = T_NODE.sub(lambda m: m.group(1) + remap(m.group(2)) + m.group(3), xml)
    after = collections.Counter(TOK.findall(''.join(m.group(2) for m in T_NODE.finditer(xml))))

    open(PATH, 'w', encoding='utf-8').write(xml)

    print('\nrenumbering (counts must be preserved):')
    ok = True
    for old, new in MAP.items():
        bcount, acount = before.get(old, 0), after.get(new, 0)
        if bcount:
            good = acount == bcount
            ok &= good
            print(f'   {old:9s} {bcount:2d}  ->  {new:9s} {acount:2d}   {"ok" if good else "MISMATCH"}')
    print('\nAll counts preserved.' if ok else '\nWARNING: mismatch.')


if __name__ == '__main__':
    main()
