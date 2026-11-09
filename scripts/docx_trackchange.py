"""
Tracked-change find/replace helper, operating at the <w:t> element level
(NOT the <w:r> level) to safely handle runs that contain more than one
<w:t> (e.g. split by a bare <w:lastRenderedPageBreak/> marker) without
silently discarding the sibling <w:t> content.

For a plain <w:t> (no enclosing w:ins): the matched <w:t> is replaced with
    <w:t>before</w:t></w:r> <w:del>...</w:del> <w:ins>...</w:ins> <w:r>{rpr}<w:t>after</w:t>
i.e. the run is closed right after "before", the del/ins pieces are their
own full runs, and a new run (same rPr) is reopened for "after" — anything
that followed the original </w:t> (e.g. <w:lastRenderedPageBreak/>, a
sibling <w:t>, the final </w:r>) is left completely untouched.

For a <w:t> already inside an unaccepted <w:ins author=A date=D>: same
idea, but the before/after reopened runs AND the del/ins pieces are each
wrapped in their own copy of <w:ins author=A date=D>, per the documented
"editing your own not-yet-accepted insertion" pattern.
"""
import re

AUTHOR = "Prasad Kulkarni"
DATE = "2026-08-02T01:00:00Z"

_ID_COUNTER = [29000]


def _next_id():
    _ID_COUNTER[0] += 1
    return _ID_COUNTER[0]


def xml_escape(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


T_ONLY_RE = re.compile(r'<w:t\b(?P<attrs>[^>]*)>(?P<text>.*?)</w:t>', re.DOTALL)
INS_DEL_TOKEN_RE = re.compile(r'<w:ins\b[^>]*>|</w:ins>|<w:del\b[^>]*>|</w:del>')
INS_OPEN_RE = re.compile(r'<w:ins\b([^>]*)>')


def _attr(attrs_str, name):
    m = re.search(rf'{name}="([^"]*)"', attrs_str)
    return m.group(1) if m else None


def _enclosing_ins_at(xml, pos):
    """Replay w:ins open/close tokens from doc start to `pos` and return the
    innermost still-open <w:ins>'s (author, date), or None. Ignores w:del
    tokens (we only care about w:ins nesting here since w:del never wraps
    plain baseline text in this document)."""
    stack = []
    for m in re.finditer(r'<w:ins\b[^>]*>|</w:ins>', xml[:pos]):
        tok = m.group(0)
        if tok.startswith('<w:ins'):
            attrs = INS_OPEN_RE.match(tok).group(1)
            stack.append({'author': _attr(attrs, 'w:author'), 'date': _attr(attrs, 'w:date')})
        else:
            if stack:
                stack.pop()
    return stack[-1] if stack else None


RPR_RE = re.compile(r'<w:rPr>.*?</w:rPr>', re.DOTALL)
R_OPEN_RE = re.compile(r'<w:r\b[^>]*>')


def _rpr_for(xml, t_start):
    """Find the <w:rPr> belonging to the run containing the <w:t> at t_start,
    even if the <w:rPr> isn't immediately adjacent (e.g. this is the 2nd
    <w:t> in a run split by a bare <w:lastRenderedPageBreak/>)."""
    run_start = None
    for m in R_OPEN_RE.finditer(xml, 0, t_start):
        run_start = m.end()
    if run_start is None:
        return ''
    window = xml[run_start:t_start]
    rm = RPR_RE.search(window)
    return rm.group(0) if rm else ''


def find_t_with_text(xml, needle, occurrence=0, exact=False):
    """Find the occurrence-th <w:t>...</w:t> (optionally exact full-text
    match) containing `needle`. Returns (t_start, t_end, rpr_xml, text,
    enclosing_ins)."""
    found = []
    for m in T_ONLY_RE.finditer(xml):
        text = m.group('text')
        match = (text == needle) if exact else (needle in text)
        if match:
            found.append((m.start(), m.end(), text))
    if not found:
        raise ValueError(f"NOT FOUND: {needle!r}")
    if occurrence >= len(found):
        raise ValueError(f"Only {len(found)} occurrence(s) of {needle!r} found, asked for #{occurrence}")
    t_start, t_end, text = found[occurrence]
    rpr = _rpr_for(xml, t_start)
    enclosing = _enclosing_ins_at(xml, t_start)
    return t_start, t_end, rpr, text, enclosing


def replace_text(xml, old, new, occurrence=0, author=AUTHOR, date=DATE, label=None, exact=False):
    try:
        t_start, t_end, rpr, text, enclosing = find_t_with_text(xml, old, occurrence, exact=exact)
    except ValueError as e:
        raise ValueError(f"{e} (label={label})")

    idx = text.index(old)
    before_txt = text[:idx]
    after_txt = text[idx + len(old):]

    def t_elem(txt):
        return f'<w:t xml:space="preserve">{xml_escape(txt)}</w:t>'

    del_id = _next_id()
    ins_id = _next_id()
    del_block = (f'<w:del w:id="{del_id}" w:author="{author}" w:date="{date}">'
                 f'<w:r>{rpr}<w:delText xml:space="preserve">{xml_escape(old)}</w:delText></w:r></w:del>')
    ins_block = (f'<w:ins w:id="{ins_id}" w:author="{author}" w:date="{date}">'
                 f'<w:r>{rpr}{t_elem(new)}</w:r></w:ins>')

    close_before = f'{t_elem(before_txt) if before_txt else ""}</w:r>'
    reopen_after = f'<w:r>{rpr}{t_elem(after_txt) if after_txt else ""}'

    if enclosing is None:
        replacement = close_before + del_block + ins_block + reopen_after
    else:
        eauth, edate = enclosing['author'], enclosing['date']

        def wrap(inner):
            wid = _next_id()
            return f'<w:ins w:id="{wid}" w:author="{eauth}" w:date="{edate}">{inner}</w:ins>'

        # before/after context text is UNCHANGED content from the original
        # (still-open) insertion, so it stays wrapped in a copy of that same
        # enclosing <w:ins>. The deleted "old" text was also part of that
        # original insertion, so it's wrapped too (nested <w:ins><w:del>,
        # per the documented pattern for editing one's own unaccepted
        # insertion). The NEW replacement text is a genuinely fresh
        # insertion authored today — it must NOT be re-wrapped in a copy of
        # the old enclosing <w:ins>, or tools will read it as "inserted on
        # the OLD date" and its later removal (rejecting today's edit) would
        # also wipe out the old del's restored text, or worse, accepting
        # today's edit leaves nothing at all (observed bug: LibreOffice/
        # pandoc treated both del and ins as removed when ins was
        # double-wrapped).
        wrapped_before = wrap(f'<w:r>{rpr}{t_elem(before_txt)}</w:r>') if before_txt else ''
        wrapped_del = wrap(del_block)
        wrapped_after = wrap(f'<w:r>{rpr}{t_elem(after_txt)}</w:r>') if after_txt else ''
        # Close the ORIGINAL run right where the <w:t> was (</w:r>), emit the
        # pieces, then reopen a bare run so any content that followed the
        # original </w:t> (lastRenderedPageBreak, sibling w:t, the final
        # </w:r>) still lands inside a valid run.
        replacement = '</w:r>' + wrapped_before + wrapped_del + ins_block + wrapped_after + f'<w:r>{rpr}'

    return xml[:t_start] + replacement + xml[t_end:]


def apply_all(xml, replacements, stop_on_error=True):
    errs = []
    applied = []
    for r in replacements:
        try:
            xml = replace_text(xml, r['old'], r['new'], occurrence=r.get('occurrence', 0),
                                label=r.get('label'), exact=r.get('exact', False))
            applied.append(r.get('label'))
        except ValueError as e:
            errs.append(str(e))
    if errs and stop_on_error:
        raise ValueError("FAILURES:\n" + "\n".join(errs) + f"\n\nApplied OK before failure: {applied}")
    return xml, errs
