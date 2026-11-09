"""
Renumber all tables into one sequence following order of first appearance,
removing the ad-hoc "H" series used for the harmonization tables.

    Table 1  (dataset constituents)         -> Table 1   (unchanged)
    Table H1 (cohort derivation)            -> Table 2
    Table H2 (missing / invalid values)     -> Table 3
    Table H3 (baseline covariates)          -> Table 4
    Table 2  (XGBoost metrics)              -> Table 5
    Table 3  (downstream impact)            -> Table 6
    Table 6  (improvement-focused recourse) -> Table 7
    Table 4  (ablation headline)            -> Table 8
    Table 5  (per-feature ablation)         -> Table 9
    Table 7  (stability vs inferential)     -> Table 10

Also corrects a pre-existing defect: Table 6 (Section 4.10) appeared before
Tables 4 and 5 (Section 4.11).

APPLIED WITHOUT TRACKED CHANGES, deliberately. Renumbering is mechanical
editorial work; marking ~25 "H1"->"2" micro-edits as tracked changes buries
the substantive revisions in noise, and because most of these labels sit
inside not-yet-accepted <w:ins> blocks the resulting nested <w:ins><w:del>
does not render with strikethrough in several common viewers, so the old
"Table H1" text reads as though it were still live. The renumbering is
disclosed in the response letter instead.

Safety properties:

  * Verified beforehand that no label is split across runs in this document
    (0 of 792 paragraphs), so a per-<w:t> replacement cannot silently miss
    one. The script re-checks this and refuses to run if that ever changes.
  * The mapping has collisions in both directions (2->5 alongside 5->9,
    3->6 alongside 6->7, 4->8 and 7->10). Each node is rewritten exactly
    once from its ORIGINAL text via a single regex pass, so no substitution
    is ever re-applied to its own output.
  * Only live <w:t> is touched. Deleted text (<w:delText>) is left as-is: it
    records content already removed and should still read as it originally did.
  * Run properties, formatting and tracked-change attribution are untouched;
    only the character data inside each <w:t> changes.
"""
import collections
import re
import sys

MAP = {
    'Table H1': 'Table 2',
    'Table H2': 'Table 3',
    'Table H3': 'Table 4',
    'Table 2': 'Table 5',
    'Table 3': 'Table 6',
    'Table 6': 'Table 7',
    'Table 4': 'Table 8',
    'Table 5': 'Table 9',
    'Table 7': 'Table 10',
    'Table 1': 'Table 1',
}

TOK = re.compile(r'Table\s+(?:H\d+|\d+)')
PARA = re.compile(r'<w:p[ >].*?</w:p>', re.DOTALL)
T_NODE = re.compile(r'(<w:t\b[^>]*>)(.*?)(</w:t>)', re.DOTALL)


def remap(text):
    return TOK.sub(lambda m: MAP.get(re.sub(r'\s+', ' ', m.group(0)), m.group(0)), text)


def assert_no_split_labels(xml):
    bad = 0
    for m in PARA.finditer(xml):
        nodes = [n for _, n, _ in (mm.groups() for mm in T_NODE.finditer(m.group(0)))]
        if not nodes:
            continue
        if len(TOK.findall(''.join(nodes))) > sum(len(TOK.findall(n)) for n in nodes):
            bad += 1
    if bad:
        raise SystemExit(
            f'ABORT: {bad} paragraph(s) contain a table label split across runs. '
            'Per-node replacement would miss them. Run merge_runs.py first.')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'v9unpacked/word/document.xml'
    with open(path, encoding='utf-8') as f:
        xml = f.read()

    assert_no_split_labels(xml)

    before = collections.Counter(TOK.findall(''.join(
        m.group(2) for m in T_NODE.finditer(xml))))

    changed = [0]

    def repl(m):
        old = m.group(2)
        new = remap(old)
        if new != old:
            changed[0] += len(
                [1 for a, b in zip(TOK.findall(old), TOK.findall(new)) if a != b])
        return m.group(1) + new + m.group(3)

    xml = T_NODE.sub(repl, xml)

    after = collections.Counter(TOK.findall(''.join(
        m.group(2) for m in T_NODE.finditer(xml))))

    with open(path, encoding='utf-8', mode='w') as f:
        f.write(xml)

    print(f'{changed[0]} table label(s) renumbered\n')
    print(f'{"old label":10s} {"n":>3s}    ->  {"new label":10s} {"n":>3s}   counts preserved')
    ok = True
    for old, new in MAP.items():
        b, a = before.get(old, 0), after.get(new, 0)
        if b:
            match = (a == b) if old != new else True
            ok &= match
            print(f'  {old:10s} {b:3d}   ->  {new:10s} {a:3d}   {"ok" if match else "MISMATCH"}')
    print('\nAll reference counts preserved.' if ok else '\nWARNING: count mismatch.')


if __name__ == '__main__':
    main()
