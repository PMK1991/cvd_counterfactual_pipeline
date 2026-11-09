"""
Final submission-readiness check across the whole package.

Extends audit.py with the classes of defect that slipped past it in earlier
rounds — each check below exists because something real got through:

  * comments surviving into a CLEAN file      (found by external review)
  * repeated *phrases*, not just single words ("trends reported in RCTs trends
    reported in published...")                (found by external review)
  * stray manual line breaks                  (found by external review)
  * missing space before a citation, space before punctuation
  * numbers verified against the canonical run artefacts, and their *unit*
    (the 63.1% was the right number with the wrong denominator)
  * cross-document consistency: every section/table the response cites must
    exist in the manuscript
  * tracked-change integrity: the co-author's edits still present

Usage: python final_check.py
"""
import collections
import glob
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit import accepted_xml, paragraphs

D = '/sessions/practical-gallant-noether/mnt/r3-round2/'
RUN = '/sessions/practical-gallant-noether/mnt/cvd_counterfactual_pipeline/fresh_cf_iterations/aggregated_results'

MS_T = D + 'Which_Counterfactuals_V17_R3_31Aug26_revPK.docx'
MS_C = D + 'Which_Counterfactuals_V17_R3_31Aug26_CLEAN.docx'
RP_T = D + 'Response_to_Reviewers_R3_31Aug26_v9_tracked.docx'
RP_C = D + 'Response_to_Reviewers_R3_31Aug26_v9_CLEAN.docx'

FAIL = []


def note(ok, label, detail=''):
    print(('  PASS  ' if ok else '  FAIL  ') + label + (f'   {detail}' if detail else ''))
    if not ok:
        FAIL.append(label)


def text_of(path):
    """Join PARAGRAPHS with newlines, not runs with nothing.

    Concatenating every <w:t> fuses paragraph boundaries and manufactures
    defects that do not exist: "Section 4." + "3.2 Data Exploration" reads as a
    reference to a non-existent "Section 4.3.2", and a reference entry ending
    "...220347" followed by "[2] Cohn..." reads as a missing space before a
    citation. Every false positive in the first run of this checker came from
    that."""
    return '\n'.join(t for t, _ in paragraphs(accepted_xml(path)))


def raw(path):
    return zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')


print('=' * 78)
print('1. CLEAN files carry no review apparatus')
print('=' * 78)
for p, lbl in [(MS_C, 'manuscript'), (RP_C, 'response')]:
    z = zipfile.ZipFile(p)
    d = raw(p)
    cparts = [n for n in z.namelist() if 'comments' in n]
    note(not cparts, f'{lbl}: no comment parts', str(cparts))
    note('commentReference' not in d and 'commentRangeStart' not in d, f'{lbl}: no comment anchors')
    note(not re.search(r'<w:ins\b', d) and not re.search(r'<w:del\b', d), f'{lbl}: no tracked changes')

print()
print('=' * 78)
print("2. Tracked files retain the co-author's edits and comment threads")
print('=' * 78)
for p, lbl, exp_ins, exp_del in [(MS_T, 'manuscript', 24, 15), (RP_T, 'response', 2, 2)]:
    d = raw(p)
    c = collections.Counter(re.findall(r'<w:(ins|del)\b[^>]*w:author="Prof\. Biswajeet Pradhan"', d))
    note(c['ins'] >= exp_ins and c['del'] >= exp_del, f'{lbl}: BP edits intact', str(dict(c)))
    z = zipfile.ZipFile(p)
    cm = z.read('word/comments.xml').decode('utf-8') if 'word/comments.xml' in z.namelist() else ''
    ids = re.findall(r'<w:comment\b[^>]*w:id="(\d+)"', cm)
    anchored = set(re.findall(r'<w:commentReference w:id="(\d+)"/>', d))
    note(all(i in anchored for i in ids), f'{lbl}: all {len(ids)} comments anchored')

print()
print('=' * 78)
print('3. Text integrity (manuscript)')
print('=' * 78)
ms = text_of(MS_T)
paras = [t for t, intbl in paragraphs(accepted_xml(MS_T)) if not intbl]

sp = [ms[m.start()-45:m.start()+10] for m in re.finditer(r'\w \.', ms)]  # ellipses excluded
note(not sp, 'no space before a full stop', repr(sp[:2]))
bad_cite = [m.group(0) for m in re.finditer(r'\w\[\d+\]', ms)]
note(not bad_cite, 'space before every citation bracket', str(bad_cite[:5]))
runon = [m.group(0) for t in paras for m in re.finditer(r'[a-z]\.[A-Z][a-z]{2}', t)]
note(not runon, 'no run-together sentences within a paragraph', str(runon[:5]))
dbl = re.findall(r'\b(\w{4,})\s+\1\b', ms)
note(not dbl, 'no repeated words', str(dbl[:5]))

# Repeated phrases: only ADJACENT repetition is a defect. The same phrase
# recurring in the Abstract and again in the Contributions is normal academic
# restatement; what broke earlier was "...trends reported in RCTs trends
# reported in published..." - the same words twice in a row.
rep = []
for t in paras:
    if re.match(r'^\[\d+\]', t):
        continue  # bibliography entries legitimately repeat conference/journal names
    w = t.split()
    for i in range(len(w) - 8):
        a = ' '.join(w[i:i + 4])
        if a == ' '.join(w[i + 4:i + 8]) and len(a) > 18:
            rep.append(a)
        # partial overlap, e.g. "X Y Z ... X Y Z" within 12 words
        for k in range(3, 6):
            seg = ' '.join(w[i:i + k])
            if len(seg) > 22 and seg in ' '.join(w[i + k:i + k + 10]):
                rep.append(seg)
note(not rep, 'no adjacent repeated phrases', str(sorted(set(rep))[:3]))

nbr = len(re.findall(r'<w:br\s*/>', raw(MS_T)))
note(nbr == 1, f'manual line breaks = {nbr} (1 expected: affiliation)')

print()
print('=' * 78)
print('4. Cross-references resolve (manuscript)')
print('=' * 78)
tbl_def = set(re.findall(r'Table ([A-Z]?\d+)\.', ms))
tbl_use = set(re.findall(r'Table ([A-Z]?\d+)\b', ms))
fig_def = set(re.findall(r'Figure (\d+)\.', ms))
fig_use = set(re.findall(r'(?:Figure|Fig\.) (\d+)', ms))
app_def = set(re.findall(r'Appendix ([A-Z]):', ms))
app_use = set(re.findall(r'Appendix ([A-Z])\b', ms))
sec_def = set(re.findall(r'^(\d+(?:\.\d+)*)[. ]', '\n'.join(paras), re.M))
sec_use = set(re.findall(r'Section (\d+(?:\.\d+)*)', ms))
for lbl, use, dfn in [('tables', tbl_use, tbl_def), ('figures', fig_use, fig_def),
                      ('appendices', app_use, app_def), ('sections', sec_use, sec_def)]:
    note(not (use - dfn), f'{lbl} all resolve', str(sorted(use - dfn)))
bib = set(re.findall(r'^\[(\d+)\]', '\n'.join(paras), re.M))
cited = set()
for t in paras:
    if not re.match(r'^\[\d+\]', t):
        cited |= set(re.findall(r'\[(\d+)\]', t))
note(not (cited - bib), 'every citation is listed', str(sorted(cited - bib, key=int)))
note(not (bib - cited), 'every reference is cited', str(sorted(bib - cited, key=int)))

print()
print('=' * 78)
print('5. Numbers match the canonical run (value AND unit)')
print('=' * 78)
import pandas as pd
scm = pd.read_csv(f'{RUN}/ci_results.csv').set_index('metric')
pb = pd.read_csv(f'{RUN}/patient_bootstrap_ci.csv').set_index('metric')
for feat, label in [('trestbps', 'Resting BP'), ('cp', 'Chest pain'), ('exang', 'Exercise angina'),
                    ('oldpeak', 'ST depression'), ('thalach', 'Max heart rate'),
                    ('slope', 'ST slope'), ('restecg', 'Resting ECG')]:
    s = scm.loc[f'{feat}_improved_pct']
    p = pb.loc[f'{feat}_improved_pct']
    stab = f"[{s['ci_lower']:.1f}%, {s['ci_upper']:.1f}%]"
    inf = f"[{p['ci_lower']:.1f}%, {p['ci_upper']:.1f}%]"
    note(stab in ms and inf in ms, f'{label}: both intervals present', f'{stab} {inf}')
fr = scm.loc['target_flip_rate_pct']
note(f"{fr['mean']:.1f}%" in ms, 'flip rate value', f"{fr['mean']:.1f}%")
note('63.1% of SCM-validated counterfactuals' in ms, 'exang 63.1% carries the counterfactual-level unit')
note('17 of 44 male patients' in ms and '4/4 = 100%' in ms, 'sex comparison on the patient-level unit')

print()
print('=' * 78)
print('6. Response letter agrees with the manuscript')
print('=' * 78)
rp = text_of(RP_T)
for m in sorted(set(re.findall(r'Section (\d+(?:\.\d+)*)', rp))):
    note(m in sec_def, f'response cites Section {m}: exists')
for m in sorted(set(re.findall(r'Table ([A-Z]?\d+)\b', rp))):
    note(m in tbl_def, f'response cites Table {m}: exists')
for m in sorted(set(re.findall(r'Appendix ([A-Z])\b', rp))):
    note(m in app_def, f'response cites Appendix {m}: exists')
note('an earlier draft compared' not in ms, 'no revision-history commentary in the manuscript')
note('reliable gender disparity' not in rp, 'response does not quote text removed from §4.9')
note('introduced by pooling' not in rp and 'introduced by the pooling' not in rp,
     'response duplicate-origin claim aligned with Appendix D')

print()
print('=' * 78)
print('7. Reviewer-flagged wording (R3 comments 1, 2, 4, 5)')
print('=' * 78)
banned = ['greater physiological capacity', 'realistic and clinically relevant',
          '63.1% of patients following', 'clinical concordance', 'reduce risk', 'improve patient']
for b in banned:
    note(b not in ms, f'removed: "{b}"')
note('k-fold' not in ms, 'k-fold not mentioned anywhere in the manuscript')

note('residual protocol heterogeneity' in rp, 'protocol heterogeneity acknowledged in the response')

print()
print('=' * 78)
print('RESULT:', 'ALL CHECKS PASS' if not FAIL else f'{len(FAIL)} FAILURE(S)')
for f in FAIL:
    print('   -', f)
print('=' * 78)
