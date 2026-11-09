"""
Reviewer 3, Comment 3 — carry patient-level uncertainty into the Methods and
into the main results table, rather than confining it to Section 4.12/Table 7.

Two additions:

  1. A new Methods subsection (3.6 Uncertainty Quantification) that defines both
     interval types and states which one to cite for population-level reading.
     Section 3 previously did not mention uncertainty quantification at all —
     both interval types were introduced ad hoc in Results.

  2. A new final column on Table 3 (downstream impact) giving the patient-level
     inferential interval alongside the algorithmic-stability interval, so the
     headline per-feature results carry the wider bounds directly. The reviewer's
     objection was that the reported intervals understate uncertainty in "every
     quantitative claim"; a standalone Table 7 does not fully answer that.

Numbers are read live from the canonical run's
aggregated_results/{ci_results.csv, patient_bootstrap_ci.csv} rather than being
hard-coded, so the table cannot drift from the artefacts.

Both are emitted as tracked changes. The new column uses <w:cellIns> in each new
cell's <w:tcPr> — the correct OOXML for an inserted table column — with the cell
text additionally wrapped in <w:ins>. This is new data the reviewer should see
marked, unlike the purely editorial renumbering, which was applied untracked.
"""
import re
import sys

import pandas as pd

AUTHOR = "Prasad Kulkarni"
DATE = "2026-08-08T00:00:00Z"
PATH = 'v9unpacked/word/document.xml'
RUN = '/sessions/practical-gallant-noether/mnt/cvd_counterfactual_pipeline/fresh_cf_iterations/aggregated_results'

_id = [71000]


def nid():
    _id[0] += 1
    return _id[0]


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


RPR = ('<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
       'w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:color w:val="111111"/>'
       '<w:sz w:val="24"/><w:szCs w:val="24"/>')
RPR_B = ('<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" '
         'w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b/><w:color w:val="111111"/>'
         '<w:sz w:val="24"/><w:szCs w:val="24"/>')

ROWS = [
    ('Resting BP (trestbps)', 'trestbps'),
    ('Chest Pain (cp)', 'cp'),
    ('Exang (1→0 / 0→1)', 'exang'),
    ('ST Depression (oldpeak)', 'oldpeak'),
    ('Max Heart Rate (thalach)', 'thalach'),
    ('ST Slope (slope)', 'slope'),
    ('Resting ECG (restecg)', 'restecg'),
]

METHODS = (
    "3.6 Uncertainty Quantification"
)

METHODS_BODY = (
    "Two distinct sources of variation are reported separately throughout the Results, because they "
    "answer different questions and differ in magnitude by roughly an order of magnitude. "
    "Algorithmic-stability intervals are 95% percentile intervals computed across 100 independent "
    "executions of the pipeline on a fixed cohort: successive runs differ only in the stochastic "
    "components of generation — the initialisation and crossover of DiCE's genetic search and the "
    "SCM's interventional sampler — while the cohort, the fitted classifier and the fitted structural "
    "model are held constant. They therefore quantify reproducibility under seed variation and say "
    "nothing about sampling variability in the underlying patient population. Patient-level "
    "inferential intervals are obtained by a patient-cluster bootstrap: the 48 analysis patients are "
    "resampled with replacement 5,000 times, every quantity is recomputed on each resampled cohort, "
    "and the 2.5th and 97.5th percentiles of the resulting distribution are reported. Resampling at "
    "the patient level rather than the counterfactual level preserves the clustering of "
    "counterfactuals within patients, which matters here because per-patient recourse feasibility is "
    "strongly bimodal (Section 4.12). The patient-level intervals are the appropriate uncertainty for "
    "any population-level reading of a result, and are the intervals we recommend citing; the "
    "algorithmic-stability intervals should be read only as reproducibility diagnostics. Both are "
    "reported side by side for the headline flip rate and each downstream feature in Section 4.12, "
    "and the per-feature results in Table 3 carry both."
)

CAPTION_ADD = (" The final column reports the patient-level inferential interval from the "
               "patient-cluster bootstrap of Section 4.12; it is the appropriate interval for a "
               "population-level reading and is substantially wider than the stability interval.")


def ins_para(text, rpr=RPR, heading=False):
    ppr = ('<w:pPr>'
           + ('<w:pStyle w:val="Heading2"/>' if heading else '')
           + '<w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/>'
           + ('<w:spacing w:before="240" w:after="120" w:line="300" w:lineRule="auto"/>'
              if heading else '<w:spacing w:after="120" w:line="360" w:lineRule="auto"/>')
           + '<w:jc w:val="both"/>'
           + f'<w:rPr><w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}"/>{rpr}</w:rPr></w:pPr>')
    return (f'<w:p>{ppr}<w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">'
            f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t>'
            f'</w:r></w:ins></w:p>')


def new_cell(text, header=False):
    """A tracked-inserted table cell (<w:cellIns> marks the column as inserted)."""
    rpr = RPR_B if header else RPR
    fill = 'D8E4F0' if header else 'FFFFFF'
    return (f'<w:tc><w:tcPr><w:tcW w:w="817" w:type="pct"/>'
            f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>'
            f'<w:cellIns w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}"/></w:tcPr>'
            f'<w:p><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>'
            f'<w:spacing w:before="180" w:after="240"/><w:jc w:val="both"/>'
            f'<w:rPr><w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}"/>{rpr}</w:rPr></w:pPr>'
            f'<w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">'
            f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t>'
            f'</w:r></w:ins></w:p></w:tc>')


def main():
    scm = pd.read_csv(f'{RUN}/ci_results.csv').set_index('metric')
    pb = pd.read_csv(f'{RUN}/patient_bootstrap_ci.csv').set_index('metric')

    values = []
    for label, feat in ROWS:
        p = pb.loc[f'{feat}_improved_pct']
        values.append(f"[{p['ci_lower']:.1f}%, {p['ci_upper']:.1f}%]")
        assert int(p['n_bootstrap']) == 5000, 'expected 5,000 bootstrap replicates'

    xml = open(PATH, encoding='utf-8').read()

    # ---------------- 1. Table 3: append a column ------------------------
    i = xml.find('Downstream impact of interventions')
    ts = xml.find('<w:tbl>', i)
    te = xml.find('</w:tbl>', ts) + len('</w:tbl>')
    tbl = xml[ts:te]

    rows = list(re.finditer(r'<w:tr\b[^>]*>.*?</w:tr>', tbl, re.DOTALL))
    if len(rows) != 8:
        raise SystemExit(f'ABORT: expected 8 rows in Table 3, found {len(rows)}')

    # widen the grid: add an 8th column, rescaling the existing ones
    grid_old = re.search(r'<w:tblGrid>.*?</w:tblGrid>', tbl, re.DOTALL).group(0)
    widths = [int(w) for w in re.findall(r'<w:gridCol w:w="(\d+)"/>', grid_old)]
    if len(widths) != 7:
        raise SystemExit(f'ABORT: expected 7 gridCols, found {len(widths)}')
    total = sum(widths)
    new_w = [1620, 880, 880, 880, 1330, 1250, 1620, 1620]
    if sum(new_w) != total:
        new_w[-1] += total - sum(new_w)
    grid_new = '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in new_w) + '</w:tblGrid>'

    out, last = [], 0
    for k, m in enumerate(rows):
        cell = new_cell('Patient-Level Inferential Interval [95%]', header=True) if k == 0 \
            else new_cell(values[k - 1])
        out.append(tbl[last:m.end() - len('</w:tr>')])
        out.append(cell)
        out.append('</w:tr>')
        last = m.end()
    out.append(tbl[last:])
    tbl_new = ''.join(out).replace(grid_old, grid_new)

    xml = xml[:ts] + tbl_new + xml[te:]

    # ---------------- 2. extend Table 3's caption -------------------------
    cap = 'Table 3. Downstream impact of interventions.'
    j = xml.find('Downstream impact of interventions.')
    if j < 0:
        raise SystemExit('ABORT: Table 3 caption not found')
    node = re.search(r'<w:t\b[^>]*>([^<]*Downstream impact of interventions\.)</w:t>', xml)
    if node:
        rpr_cap = RPR_B
        add = (f'</w:r><w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">'
               f'<w:r><w:rPr>{rpr_cap}</w:rPr>'
               f'<w:t xml:space="preserve">{esc(CAPTION_ADD)}</w:t></w:r></w:ins>'
               f'<w:r><w:rPr>{rpr_cap}</w:rPr>')
        end = node.end()
        xml = xml[:end] + add + xml[end:]

    # ---------------- 3. Methods subsection 3.6 ---------------------------
    anchor = xml.find('4. Results')
    if anchor < 0:
        raise SystemExit('ABORT: could not find the Results heading')
    pstart = max(xml.rfind('<w:p ', 0, anchor), xml.rfind('<w:p>', 0, anchor))
    block = ins_para(METHODS, rpr=RPR_B, heading=True) + ins_para(METHODS_BODY)
    xml = xml[:pstart] + block + xml[pstart:]

    open(PATH, 'w', encoding='utf-8').write(xml)

    print('Table 3: added patient-level interval column')
    for (label, _), v in zip(ROWS, values):
        s = scm.loc[f'{ROWS[[r[0] for r in ROWS].index(label)][1]}_improved_pct']
        print(f'   {label:26s} {s["mean"]:5.1f}  [{s["ci_lower"]:.1f}%, {s["ci_upper"]:.1f}%]   {v}')
    print('\nAdded Methods subsection 3.6 Uncertainty Quantification')
    print('Extended Table 3 caption')


if __name__ == '__main__':
    main()
