"""
Renumber all tables into a single sequence following order of first appearance,
removing the ad-hoc "H" series introduced for the harmonization tables.

    Table 1  (dataset constituents)        -> Table 1   (unchanged)
    Table H1 (cohort derivation)           -> Table 2
    Table H2 (missing / invalid values)    -> Table 3
    Table H3 (baseline covariates)         -> Table 4
    Table 2  (XGBoost metrics)             -> Table 5
    Table 3  (downstream impact)           -> Table 6
    Table 6  (improvement-focused recourse)-> Table 7
    Table 4  (ablation headline)           -> Table 8
    Table 5  (per-feature ablation)        -> Table 9
    Table 7  (stability vs inferential)    -> Table 10

This also corrects a pre-existing defect: Table 6 (Section 4.10) appeared
before Tables 4 and 5 (Section 4.11).

The mapping has collisions in both directions (2->5 and 5->9, 3->6 and 6->7,
4->8 and 7->10), so replacement is done node-by-node in a single pass over the
ORIGINAL text of each <w:t>, walking the document in reverse so byte offsets
stay valid and no substitution is ever re-applied to its own output.

Text inside <w:delText> is deliberately left alone: it is a record of content
already removed and should continue to read as it originally did.

Every change is emitted as a tracked deletion + insertion; where the text sits
inside a not-yet-accepted <w:ins>, the deletion is nested inside a copy of that
insertion and the new text is emitted as a fresh insertion of its own (see
trackchange2 for why the new text must NOT be re-wrapped).
"""
import re
import sys

sys.path.insert(0, '.')
import trackchange2 as tc

AUTHOR = "Prasad Kulkarni"
DATE = "2026-08-08T00:00:00Z"

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

TOK = re.compile(r'Table (?:H\d+|\d+)')
T_NODE = re.compile(r'<w:t\b[^>]*>(.*?)</w:t>', re.DOTALL)

_id = [51000]


def nid():
    _id[0] += 1
    return _id[0]


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def remap(s):
    return TOK.sub(lambda m: MAP.get(m.group(0), m.group(0)), s)


def build_replacement(xml, start, old_text, new_text):
    """Tracked del+ins replacing the whole <w:t> node at `start`."""
    rpr = tc._rpr_for(xml, start)
    enclosing = tc._enclosing_ins_at(xml, start)

    del_block = (f'<w:del w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">'
                 f'<w:r>{rpr}<w:delText xml:space="preserve">{esc(old_text)}'
                 f'</w:delText></w:r></w:del>')
    ins_block = (f'<w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">'
                 f'<w:r>{rpr}<w:t xml:space="preserve">{esc(new_text)}'
                 f'</w:t></w:r></w:ins>')

    if enclosing is not None:
        eauth, edate = enclosing['author'], enclosing['date']
        del_block = (f'<w:ins w:id="{nid()}" w:author="{eauth}" w:date="{edate}">'
                     f'{del_block}</w:ins>')

    return '</w:r>' + del_block + ins_block + f'<w:r>{rpr}'


def main():
    path = 'v9unpacked/word/document.xml'
    with open(path, encoding='utf-8') as f:
        xml = f.read()

    hits = []
    for m in T_NODE.finditer(xml):
        old = m.group(1)
        new = remap(old)
        if new != old:
            hits.append((m.start(), m.end(), old, new))

    print(f'{len(hits)} text node(s) contain a table reference to renumber')
    for _, _, o, n in hits:
        a = TOK.findall(o)
        b = TOK.findall(n)
        print(f'   {", ".join(a)}  ->  {", ".join(b)}      | {o[:58]!r}')

    for start, end, old, new in reversed(hits):
        xml = xml[:start] + build_replacement(xml, start, old, new) + xml[end:]

    with open(path, 'w', encoding='utf-8') as f:
        f.write(xml)
    print(f'\nApplied {len(hits)} tracked renumbering edit(s).')


if __name__ == '__main__':
    main()
