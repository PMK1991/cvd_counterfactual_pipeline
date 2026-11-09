"""
Build a genuinely submission-ready CLEAN docx: all tracked changes accepted AND
all comments removed.

The earlier clean build only accepted tracked changes, so both CLEAN files still
carried Word comments (4 in the manuscript, 8 in the response) - review
apparatus that must not reach the journal.

Accepting changes:
  * paired <w:ins>...</w:ins>  -> keep inner content
  * paired <w:del>...</w:del>  -> drop entirely
  * self-closing <w:ins/>      -> drop the marker (an INSERTED paragraph mark;
                                  the mark itself stays)
  * self-closing <w:del/>      -> asserted absent; it means a DELETED paragraph
                                  mark, i.e. "merge with the following
                                  paragraph", which cannot simply be dropped

Removing comments requires clearing six things, or Word reports the file as
corrupt: the anchors and reference runs in document.xml, the four comment parts,
their relationship entries, and their [Content_Types] overrides.
"""
import os
import re
import shutil
import sys
import zipfile

DEL = re.compile(r'<w:del\b[^>]*>((?:(?!<w:ins\b|<w:del\b).)*?)</w:del>', re.DOTALL)
INS = re.compile(r'<w:ins\b[^>]*>((?:(?!<w:ins\b|<w:del\b).)*?)</w:ins>', re.DOTALL)

COMMENT_PARTS = ('word/comments.xml', 'word/commentsExtended.xml',
                 'word/commentsIds.xml', 'word/commentsExtensible.xml')


def accept_changes(xml):
    cur = xml
    while True:
        n = DEL.sub('', cur)
        n = INS.sub(r'\1', n)
        if n == cur:
            break
        cur = n
    if re.search(r'<w:del\b[^>]*/>', cur):
        raise SystemExit('ABORT: self-closing <w:del/> found (deleted paragraph mark) - needs a merge, not a drop')
    return re.sub(r'<w:ins\b[^>]*/>', '', cur)


def strip_comments(xml):
    # ranges and the runs that carry the reference marks
    xml = re.sub(r'<w:commentRangeStart\b[^>]*/>', '', xml)
    xml = re.sub(r'<w:commentRangeEnd\b[^>]*/>', '', xml)
    xml = re.sub(r'<w:r\b[^>]*>(?:(?!</w:r>).)*?<w:commentReference\b[^>]*/>.*?</w:r>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'<w:commentReference\b[^>]*/>', '', xml)
    return xml


def main(src, dst):
    zin = zipfile.ZipFile(src)
    names = [n for n in zin.namelist() if n not in COMMENT_PARTS]
    parts = {n: zin.read(n) for n in names}

    doc = parts['word/document.xml'].decode('utf-8')
    doc = strip_comments(accept_changes(doc))
    parts['word/document.xml'] = doc.encode('utf-8')

    # drop relationships pointing at the removed comment parts
    rels = parts.get('word/_rels/document.xml.rels', b'').decode('utf-8')
    if rels:
        rels = re.sub(r'<Relationship\b[^>]*Target="comments(?:Extended|Ids|Extensible)?\.xml"[^>]*/>', '', rels)
        parts['word/_rels/document.xml.rels'] = rels.encode('utf-8')

    # drop the content-type overrides for them
    ct = parts.get('[Content_Types].xml', b'').decode('utf-8')
    if ct:
        ct = re.sub(r'<Override\b[^>]*PartName="/word/comments(?:Extended|Ids|Extensible)?\.xml"[^>]*/>', '', ct)
        parts['[Content_Types].xml'] = ct.encode('utf-8')

    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, parts[n])

    check = zipfile.ZipFile(dst)
    d = check.read('word/document.xml').decode('utf-8')
    n_ins = len(re.findall(r'<w:ins\b', d))
    n_del = len(re.findall(r'<w:del\b', d))
    n_ref = len(re.findall(r'commentReference|commentRangeStart', d))
    left = [p for p in COMMENT_PARTS if p in check.namelist()] or 'none'
    print(os.path.basename(dst))
    print(f'   tracked changes : ins={n_ins} del={n_del}')
    print(f'   comment parts   : {left}')
    print(f'   comment refs    : {n_ref}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
