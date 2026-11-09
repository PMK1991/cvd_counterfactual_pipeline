"""
Build the Response to Reviewers document for the R3 round, matching the tone and
formatting of the archived reviewer response under
docs/manuscripts/reviewer-r3/Final Submission/:

  - Title line, underlined+bold
  - Manuscript ID / Title / Journal metadata block
  - "Dear Editor," + framing paragraph
  - Underlined Heading-1 style section banners per reviewer
  - Bold "Comment N", reviewer text in italics (VERBATIM, never shortened)
  - Bold "Authors' Response:" lead-in, plain body
  - Closing Statement
"""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

FONT = 'Times New Roman'
SIZE = Pt(12)


def _style_run(r, bold=False, italic=False, underline=False, size=SIZE):
    r.font.name = FONT
    r.font.size = size
    r.bold = bold
    r.italic = italic
    r.underline = underline
    r.font.color.rgb = RGBColor(0, 0, 0)
    return r


def para(doc, text='', bold=False, italic=False, underline=False,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=SIZE, space_after=Pt(10)):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = space_after
    p.paragraph_format.line_spacing = 1.15
    if text:
        _style_run(p.add_run(text), bold=bold, italic=italic, underline=underline, size=size)
    return p


def lead(doc, label, body):
    """Bold lead-in label followed by plain body text in the same paragraph."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.15
    _style_run(p.add_run(label), bold=True)
    _style_run(p.add_run(body))
    return p


def banner(doc, text):
    """Underlined bold section banner, mirroring the template's Heading-1 look."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(10)
    _style_run(p.add_run(text), bold=True, underline=True, size=Pt(14))
    return p


def quote(doc, text):
    """Verbatim reviewer text, italicised."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.15
    _style_run(p.add_run(text), italic=True)
    return p


doc = Document()
st = doc.styles['Normal']
st.font.name = FONT
st.font.size = SIZE

# ---------------------------------------------------------------- header
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
_style_run(p.add_run('Response to Reviewers'), bold=True, underline=True, size=Pt(14))

lead(doc, 'Manuscript ID: ', 'd4fb8c42-11e8-4b74-9650-c8be34640996')
lead(doc, 'Manuscript Title: ',
     'Counterfactual Recourse with Structural Causal Models for Cardiovascular Risk Mitigation')
lead(doc, 'Journal: ', 'Scientific Reports')

para(doc, 'Dear Editor,', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)

para(doc,
     'We thank the Editor and Reviewer 3 for continuing to engage so carefully with this manuscript. '
     'We are grateful for the explicit acknowledgement of the improvements made across the previous '
     'rounds, and we have treated the remaining concerns as the substantive methodological objections '
     'they are rather than as presentational matters. In this revision we have done three things. '
     'First, where a concern is foundational to the cross-sectional study design — Comments 1 and 2 — '
     'we have not attempted to argue it away; we have instead narrowed the manuscript\'s claims until '
     'they sit strictly inside what the data can support, and have said so plainly in the Abstract, '
     'Results and Limitations. Second, where a concern identified a quantitative gap that could be '
     'closed — Comments 3, 4 and 6 — we have closed it with new analyses and new tables rather than '
     'with additional hedging language. Third, where we respectfully disagree with a specific '
     'recommendation, we have set out our reasoning and the supporting arithmetic explicitly rather '
     'than passing over it in silence; this applies only to the k-fold cross-validation suggestion in '
     'Comment 3, and we would implement it without hesitation should the Editor consider it necessary.')

para(doc,
     'The principal additions in this revision are: a patient-level cluster bootstrap and a new '
     'Table 7 that report inferential uncertainty alongside the algorithmic-stability intervals, '
     'adopting the reviewer\'s own framing of the distinction (Section 4.12); a new Table D3 giving '
     'the full baseline covariate distributions across the three nested cohorts, which directly '
     'addresses whether the 48-patient cohort constitutes a coherent clinical population '
     '(Section 3.1.1); a new improvement-focused recourse analysis that separates classifier '
     'boundary-crossing from graded physiological movement (Section 4.10, Table 4); and a full '
     'per-patient decomposition of the flip rate with the accompanying Figure 13. All numerical '
     'results throughout the manuscript have additionally been regenerated from a single, '
     'deterministically seeded 100-iteration run so that every quoted figure is traceable to one '
     'reproducible execution. All substantive changes are marked in the tracked-change version of the '
     'manuscript. References below are given by section, table, figure and appendix rather than by page '
     'number, since pagination differs between the tracked and clean views of the file. Two purely editorial changes are not marked as tracked edits, because '
     'marking several dozen label changes would have obscured the substantive revisions without '
     'conveying anything of scientific content. First, the dataset-cleansing narrative and its '
     'three tables have been relocated from Section 3.1.1 to a new Appendix D (Dataset Cleansing '
     'and Cohort Derivation), continuing the manuscript\'s existing Appendix A–C series; the '
     'content is unchanged, and Section 3.1.1 retains the dataset provenance description with a '
     'forward pointer. Second, the main-text tables have consequently been renumbered into a '
     'single sequence in order of appearance (Tables 1–7), with the appendix tables labelled '
     'D1–D3 to match the existing B1/B2 convention. This also corrects a pre-existing '
     'inconsistency in which the table of Section 4.10 was numbered after those of Section 4.11.')

# ---------------------------------------------------------------- Reviewer 1
banner(doc, 'Response to Reviewer 1')

para(doc, 'Comment 1', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
quote(doc, 'None.')
lead(doc, 'Authors\' Response: ',
     'We thank Reviewer 1 for indicating that no further concerns remain following the previous '
     'round of revisions.')

# ---------------------------------------------------------------- Reviewer 3
banner(doc, 'Response to Reviewer 3')

para(doc, 'General Comment', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
quote(doc,
      'We thank the authors for their submission and acknowledge the genuine improvements made across '
      'revision rounds, particularly the reframing of the SCM as a domain-knowledge prior, the addition '
      'of the ablation study, and the improved cohort documentation. The research question is important '
      'and the integration of XAI with causal reasoning is a worthwhile direction. However, several '
      'fundamental concerns remain that prevent acceptance in the current form.')
lead(doc, 'Authors\' Response: ',
     'We thank the reviewer for this assessment and for recognising the direction of travel across the '
     'revision rounds. We accept the framing of the remaining concerns and have separated them into two '
     'categories in the revision. Comments 1 and 2 identify limitations that are structural to a '
     'cross-sectional dataset and cannot be removed by any analysis performed on these data; for these '
     'we have narrowed the claims rather than defended them, and the manuscript now states explicitly '
     'what the design can and cannot support. Comments 3, 4, 5 and 6 identify gaps that are addressable, '
     'and we have addressed each with new quantitative work reported in the manuscript rather than '
     'deferring any of it. We set out below, comment by comment, exactly what was done and where it '
     'appears.')

# ---- Comment 1
para(doc, 'Comment 1', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
quote(doc, '1. Foundational Data-Design Mismatch')
quote(doc,
      'The central clinical question — which cholesterol-lowering intervention would most effectively '
      'reduce an individual\'s CVD risk — requires causal identification of P(Y | do(chol = c′)). The '
      'cross-sectional design, with no observed treatment variation, no longitudinal follow-up, and no '
      'randomization mechanism, cannot support this identification regardless of the sophistication of '
      'the analytical pipeline.')
quote(doc,
      'The revised framing as a hypothesis-generating proof-of-concept is appreciated but does not '
      'resolve this structural limitation. As Reviewer 3 noted, this is inherent to the data structure '
      'rather than addressable through revision.')
lead(doc, 'Authors\' Response: ',
     'We agree with the reviewer without qualification. P(Y | do(chol = c′)) is not identifiable from '
     'this dataset. There is no within-patient treatment variation, no follow-up, and no randomisation, '
     'and no amount of methodological machinery applied downstream can recover identification that the '
     'design does not provide. We accept that this is a property of the data rather than of the '
     'exposition, and we have not attempted to argue otherwise in the revision.')
para(doc,
     'What we have done instead is remove the residual language that implied clinical utility beyond '
     'what the design supports, and state the limitation as a first-order concession rather than a '
     'closing caveat. Limitation (i) in Section 5.2 now states directly that the dataset is '
     'cross-sectional, that P(Y | do(chol = c′)) is therefore not identifiable from these data alone, '
     'and that the percentages reported throughout Section 4 are cohort-aggregate descriptive summaries '
     'of the simulation\'s outputs under the SCM\'s stated structural assumptions — explicitly not '
     'estimates of P(Y | do(chol = c′)) on any broader population. The Abstract now closes by '
     'stating that the findings "should not be interpreted as evidence of causal treatment effects but '
     'rather as simulations under the assumptions encoded in the Structural Causal Model," and describes '
     'the outputs as "actionable, personalized algorithmic recourse candidates for subsequent clinical '
     'evaluation" rather than as risk reductions.')
para(doc,
     'We would offer one clarification about what the contribution is now claimed to be, since we think '
     'the reviewer\'s objection and our revised claim may no longer be in conflict. We do not claim to '
     'answer which intervention would most effectively reduce an individual\'s risk. The claim is '
     'narrower: given a deployed classifier and an explicitly stated structural prior, the framework '
     'identifies which recourse candidates are internally coherent with that prior and which are not, '
     'and the ablation in Section 4.11 quantifies how much that filtering changes the recommendations. '
     'That is a statement about the behaviour of a decision-support pipeline under stated assumptions, '
     'and it is verifiable on cross-sectional data. Whether the recourse it surfaces would reduce actual '
     'risk is precisely the question we now state the data cannot answer, and which we identify as '
     'requiring prospective validation.')

# ---- Comment 2
para(doc, 'Comment 2', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
quote(doc, '2. Circular Validation')
quote(doc,
      'The SCM\'s structural equations are learned from the same cross-sectional dataset used to '
      'evaluate recourse recommendations, and the classifier is trained on the same data distribution:')
quote(doc, 'D_train → f̂_w and D_train → SCM')
quote(doc,
      'The reported clinical concordance with RCT benchmarks is therefore partially circular — the SCM '
      'absorbs associations from the data rather than from the independent RCT evidence it is compared '
      'against. Independent validation on a held-out external dataset (e.g. MIMIC-IV, UK Biobank) is '
      'necessary before concordance claims can be made.')
lead(doc, 'Authors\' Response: ',
     'We accept the substance of this concern and have removed the concordance claim in the form the '
     'reviewer objects to. The manuscript no longer presents agreement with trial benchmarks as '
     'validation of the framework. It is now described as a directional plausibility check only, and '
     'the text states that such agreement cannot constitute independent evidence because the structural '
     'model absorbs associations from the same distribution against which the comparison is drawn.')
para(doc,
     'We would, however, record one point of fact that we believe narrows the concern without '
     'dissolving it. Both the classifier and the SCM are fitted offline on the 565-record training split '
     'only, and the SCM is loaded read-only at inference time; the 142-record test split from which the '
     '48-patient recourse cohort is drawn is not seen by either model during fitting. The recourse '
     'evaluation is therefore not circular in the narrow sense of test-set leakage, and Section 3.1.1 '
     'and Figure 9 now document this fitting sequence explicitly. We nonetheless accept that this does '
     'not answer the reviewer\'s actual objection, which concerns the shared data-generating '
     'distribution rather than the shared records: a structural model estimated from one split of an '
     'observational dataset still encodes that dataset\'s associations, and comparing its outputs to '
     'external trial results does not test it against genuinely independent evidence. That objection '
     'stands, and we have written the manuscript accordingly.')
para(doc,
     'Section 5.1 now bounds the concordance discussion quantitatively rather than rhetorically. Where '
     'the modelled reduction in resting blood pressure is compared to trial estimates, the text states '
     'that the patient-level inferential interval for the proportion of counterfactuals showing reduced '
     'resting blood pressure spans [37.3%, 81.4%] (Table 7), so the cohort mean is compatible with a '
     'trial-scale effect without constituting evidence for one. Resting ECG is identified as the least '
     'corroborated of the downstream outcomes. We have also made the circularity explicit in the '
     'ablation itself: the no-SCM arm in Section 4.11 is stated to be intentionally circular, since '
     'DiCE optimised those candidates against the same classifier that then scores them, and it is '
     'therefore reported as a model-only acceptance upper bound rather than as an independent '
     'comparator. We agree that external validation on MIMIC-IV or UK Biobank is the necessary next '
     'step before any concordance claim can be made, and this is now named as such in Section 5.2 and '
     'in the Conclusion rather than listed among generic future directions.')

# ---- Comment 3
para(doc, 'Comment 3', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
quote(doc, '3. Severely Limited and Unrepresentative Analysis Cohort')
quote(doc,
      '96% of the available data is discarded before the primary analysis, leaving n = 48 true-positive '
      'test-set patients. The true inferential uncertainty around the primary flip rate estimate:')
quote(doc, 'R̂ = 34.8%, ĈI_inferential ≈ [21%, 48%]')
quote(doc,
      'is roughly ten times wider than the algorithmic-stability interval [33%, 37%] reported throughout '
      'the paper. The reported intervals systematically underrepresent the actual uncertainty in every '
      'quantitative claim.')
quote(doc,
      'We strongly recommend the authors consider k-fold cross-validation, which would increase the '
      'effective recourse cohort to approximately n ≈ 629 true-positive patients and substantially '
      'improve the precision of all estimates.')
lead(doc, 'Authors\' Response: ',
     'We agree with the diagnosis and have adopted the reviewer\'s framing directly. The revised '
     'manuscript contains a new Section 4.12 and a new Table 7 reporting a patient-cluster '
     'bootstrap in which the 48 patients are resampled with replacement 5,000 times and every quantity '
     'is recomputed on the resampled cohort. The methodology is now specified in a new Methods '
     'subsection (Section 3.6, Uncertainty Quantification), which defines both interval types, '
     'explains why resampling at the patient rather than the counterfactual level is the '
     'appropriate unit given the clustering of counterfactuals within patients, and states that '
     'the patient-level intervals are the ones we recommend citing. We accept that the previous '
     'draft introduced both interval types only in the Results and left the reader to infer the '
     'distinction. The result confirms the reviewer\'s estimate closely: the '
     'patient-level 95% interval for the flip rate is [22.1%, 47.2%], against an algorithmic-stability '
     'interval of [32.7%, 36.8%] — approximately ten times wider, as anticipated. Table 7 now reports '
     'both intervals side by side for the flip rate and for every downstream feature, and the text '
     'states that the patient-level intervals are the appropriate uncertainty for any population-level '
     'interpretation and that all quantitative claims in the paper should be read with these wider '
     'bounds in mind. Acting on the reviewer\'s point that the reported intervals understate '
     'uncertainty in every quantitative claim, we have also carried the patient-level intervals '
     'into the main results table itself: Table 3 now reports them in a final column beside the '
     'algorithmic-stability intervals, so the headline per-feature results carry the wider bounds '
     'directly rather than requiring a cross-reference to a separate table. We prefer to show the '
     'contrast plainly — resting ECG improvement, for example, is [35.5%, 40.7%] on the stability '
     'interval but [16.3%, 61.5%] once patient-sampling uncertainty is admitted.')
para(doc,
     'We also examined why the interval is so wide, since this turned out to be informative rather than '
     'merely unfortunate. The width is not driven by noise in the generation procedure but by genuine '
     'between-patient heterogeneity: of the 48 patients, 27 never achieve a flip in any of the 100 '
     'iterations and 13 achieve one in every iteration, leaving only 8 intermediate cases. The cohort '
     'therefore separates almost entirely into recourse-infeasible and recourse-feasible groups under '
     'cholesterol-only intervention. We verified that this is a property of the patients rather than of '
     'the generator by computing a dispersion index for each of the 8 intermediate patients (observed '
     'run-to-run variance in successes divided by the binomial-expected variance); the median is 1.09 '
     '(interquartile range 0.93–1.21), and an independently generated second 100-iteration run gives a '
     'consistent median of 0.93 [0.87, 1.17]. Values close to one indicate that the run-to-run variation '
     'reflects which candidates the generator happens to propose rather than unstable structural-model '
     'verdicts. This decomposition is reported in Section 4.12 and visualised in the new Figure 13.')
para(doc,
     'On the specific recommendation to adopt k-fold cross-validation, we investigated it quantitatively '
     'before deciding, and we would like to set out what we found, because our arithmetic differs from '
     'the reviewer\'s in a way that materially affects the cost-benefit assessment. We believe the '
     'figure of n ≈ 629 corresponds to the count of high-risk records in the pooled raw data rather than '
     'to out-of-fold true positives after cleaning. The cleaned dataset contains 707 records of which '
     '335 are CVD-positive in total, so 629 true positives is not reachable under any cross-validation '
     'scheme — it exceeds the number of positive records that exist. Re-running the classifier under '
     'stratified k-fold with every record held out exactly once and scored by a model that never saw it, '
     'we obtain 287 out-of-fold true positives at 5-fold and 292 at 10-fold. This is a real and useful '
     'enlargement — approximately six-fold — and it would narrow the flip-rate interval from '
     '[22.1%, 47.2%] to approximately [29.3%, 40.3%].')
para(doc,
     'We elected not to adopt it in this round, for three reasons that we offer for the reviewer\'s '
     'consideration rather than as a settled position. First, it changes the estimand: a flip rate '
     'pooled across five folds averages over five different classifier-and-SCM pairs, whereas the '
     'quantity the paper reports describes the behaviour of one deployed model, which is the object a '
     'reader would be evaluating for deployment. Second, and more importantly, a k-fold recourse run '
     'must refit the SCM within each fold as well as the classifier; refitting only the classifier '
     'would reintroduce precisely the circularity raised in Comment 2, since the structural model would '
     'then have seen the held-out fold. Third, the precision gain does not alter any conclusion the '
     'revised manuscript draws, because we have already withdrawn the population-level claims that '
     'greater precision would have supported — the wide interval is now reported as the honest '
     'uncertainty rather than treated as a deficiency to be engineered away. That said, we recognise '
     'this is a judgement call on which reasonable reviewers may differ, and we would be glad to '
     'implement the 5-fold analysis with per-fold SCM refitting if the Editor considers it necessary '
     'for acceptance.')
para(doc,
     'On the characterisation of the cohort as unrepresentative, we agree, and we have now documented '
     'this rather than left it to inference. The new Table D3 reports baseline covariate '
     'distributions across all three nested cohorts and shows that the recourse cohort is older '
     '(57.1 vs 52.8 years), more predominantly male (91.7% vs 76.4%), and carries a substantially '
     'heavier ischaemic burden than the pooled dataset. The accompanying text states that the recourse '
     'results describe this specific phenotype and should not be extrapolated to the broader CVD '
     'population. We would note that the reduction from 707 to 48 is not primarily data discarding but '
     'the definition of the analysis: recourse is only meaningful for patients the deployed model flags '
     'as high-risk, and evaluating it on held-out data restricts this to the test split. The full '
     'accounting is given step by step in Table D1.')

# ---- Comment 4
para(doc, 'Comment 4', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
quote(doc, '4. Unsupported Demographic Conclusions')
quote(doc, 'The headline demographic finding:')
quote(doc, 'R̂_female = 100%, n_f ≈ 10–15')
quote(doc,
      'is statistically uninformative at this sample size. The Clopper-Pearson 95% confidence interval '
      'at n_f = 10 is approximately [69%, 100%], rendering the gender disparity conclusion evidentially '
      'empty. No formal test or confidence interval is reported for any demographic comparison.')
quote(doc,
      'These findings should either be withdrawn or clearly labeled as preliminary observations '
      'requiring confirmation in a larger sample.')
lead(doc, 'Authors\' Response: ',
     'We agree entirely and have implemented the reviewer\'s recommendation. We should also correct the '
     'record in a direction that strengthens rather than weakens the reviewer\'s point: the female '
     'subgroup in the true-positive recourse cohort is n = 4, not 10–15. The situation is therefore '
     'worse than the reviewer\'s estimate assumed, and the corresponding Clopper-Pearson 95% confidence '
     'interval for 4 successes out of 4 is [39.8%, 100%] — an interval spanning almost the entire '
     'probability range, and consistent with a true female flip rate anywhere from below half to '
     'certainty.')
para(doc,
     'Section 4.9 has been rewritten accordingly. It now states that the comparison is '
     'exploratory and statistically underpowered, reports the exact subgroup size and the '
     'Clopper-Pearson interval, and states explicitly that we do not interpret the pattern as a reliable '
     'gender disparity and that no strong claim of clinically favourable reclassification for female '
     'patients is warranted on these data. The section notes that any apparent difference may reflect '
     'small-sample noise, differences in baseline risk, or model bias. The corresponding claims have '
     'been removed from the fairness discussion and the Conclusion. The age comparison in the same '
     'section is similarly relabelled: the text now states that age is an immutable feature that was '
     'not intervened upon, that no formal statistical test was applied, and that the descriptive '
     'pattern is exploratory.')
para(doc,
     'The new Table D3 also explains the origin of the small subgroup, which we think is worth '
     'making explicit. The recourse cohort is 91.7% male against 76.4% in the pooled dataset, because '
     'restricting to model-flagged, correctly predicted high-risk patients concentrates the male-skewed '
     'high-severity profiles. The n = 4 female subgroup is therefore a structural consequence of the '
     'cohort definition rather than an incidental sampling accident, and it will not be remedied by '
     'anything short of a larger and more gender-balanced source population. We state this in '
     'Section 3.1.1 and cross-reference it from Section 4.9.')

# ---- Comment 5
para(doc, 'Comment 5', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
quote(doc, '5. Outcome Conflation')
quote(doc, 'The paper conflates two fundamentally distinct quantities throughout:')
quote(doc,
      'f̂_w(x′_i) = 0 (classifier boundary crossing) ≠ Y_i^future = 0 (actual CVD risk reduction)')
quote(doc,
      'While the limitations section acknowledges this distinction, the abstract and introduction '
      'continue to imply clinical utility beyond what classifier boundary-crossing can support. We '
      'recommend the authors more consistently apply the hypothesis-generating framing throughout the '
      'entire manuscript, including the abstract.')
lead(doc, 'Authors\' Response: ',
     'We accept this and have applied the framing consistently rather than confining it to the '
     'Limitations section. The Abstract has been revised so that the framework is described as '
     'surfacing "actionable, personalized algorithmic recourse candidates for subsequent clinical '
     'evaluation," as informing "tailored, hypothesis-generating risk-reduction strategies that would '
     'require prospective clinical validation before use," and it now closes with an explicit sentence '
     'stating that the findings should not be interpreted as evidence of causal treatment effects but '
     'as simulations under the assumptions encoded in the Structural Causal Model. The Introduction has '
     'been revised in the same direction, and throughout the manuscript "reduce risk" has been replaced '
     'by "model-predicted risk" and "validated" by "evaluated" wherever the earlier wording implied '
     'that a boundary crossing constituted a clinical outcome.')
para(doc,
     'Beyond the language, we have added an analysis that we believe addresses the underlying concern '
     'more directly than terminology can. The reviewer\'s point is that a binary boundary crossing is '
     'not a clinical outcome. The new Section 4.10 and Table 4 therefore report '
     'improvement-focused recourse, which asks a different question: among the counterfactuals whose '
     'structural propagation did not flip the predicted label, how far and in what direction do the '
     'downstream symptoms nevertheless move? On average 155.4 counterfactuals per iteration fall into '
     'this non-flip set. Every one of them improves at least one downstream symptom, 36.5% '
     '[35.0, 38.0] improve at least one with none worsening, and the mean net gain is +1.4 [1.4, 1.5] '
     'symptoms out of six scored. The per-symptom breakdown is also informative about the limits of the '
     'framework rather than only its strengths: the continuous symptoms improve in most non-flip cases, '
     'while chest pain and exercise-induced angina register essentially no improvement and remain at '
     'their most severe modal categories, because they are children of the disease node and shift only '
     'when the predicted disease state itself changes.')
para(doc,
     'We report this because it makes the distinction the reviewer draws visible in the results rather '
     'than only in the caveats: a graded, partial physiological movement is a different object from a '
     'label flip, and the manuscript now measures both and keeps them separate. We continue to state, '
     'in Section 5.2, that neither quantity is an observed clinical outcome and that Y_future is not '
     'available in this dataset.')

# ---- Comment 6
para(doc, 'Comment 6', bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
quote(doc, '6. Missing Data and Harmonization Infrastructure')
quote(doc,
      'The paper merges five datasets collected under different protocols without documented '
      'harmonization, reports no formal missing data analysis, and provides no Table 1 summarizing '
      'covariate distributions. The de facto inclusion criteria differ across sub-cohorts but are never '
      'reconciled. This makes it impossible to assess whether the 48-patient recourse cohort represents '
      'any coherent clinical population.')
lead(doc, 'Authors\' Response: ',
     'We thank the reviewer for this comment, which prompted us to correct a factual imprecision in our '
     'own description as well as to add the missing documentation. Section 3.1.1 has been rewritten, and '
     'the record-level cleansing detail it previously carried now forms a new Appendix D containing '
     'three tables.')
para(doc,
     'On harmonization, we should clarify a point that our earlier wording obscured. The five cohorts '
     'were not merged by us. The dataset is distributed in already-harmonized form by its original '
     'curators [27], and the cross-source harmonization — retaining a common set of twelve variables '
     'across all sources, expressing them in identical units, applying identical categorical coding, '
     'and collapsing the disease label to a single binary target — was performed as part of that public '
     'release. Our earlier draft described this in the active voice and thereby implied we had '
     'performed it; this has been corrected, and Section 3.1.1 now states explicitly that the '
     'harmonization was performed by the dataset\'s original curators and that we take the distributed '
     'file as the starting point for our own processing. We think this distinction matters for the '
     'reviewer\'s concern, because it locates the residual protocol heterogeneity upstream of this '
     'paper: the variables are consistently defined in the distributed file, but we cannot reconcile '
     'collection protocols we did not observe, and we now say so.')
para(doc,
     'On missing data, Table D2 reports the missing and physiologically invalid count for every '
     'one of the twelve variables, including those with none, together with the action taken. Missing '
     'values are encoded as zero in the source data and were handled explicitly rather than imputed: '
     '172 records with cholesterol of 0 mg/dL were treated as missing and removed, which also '
     'eliminated the single record with a resting blood pressure of 0 mmHg; 272 exact duplicates '
     'introduced by the pooling of overlapping sources were dropped; and 39 multivariate outliers were '
     'removed by an inter-quartile-range rule. Oldpeak is noted as a case where zero is a valid '
     'measurement and was retained. Table D1 gives the complete cohort derivation as an '
     'auditable funnel from 1,190 pooled records to the 48-patient analysis cohort, with the records '
     'entering, removed and remaining at each step, taken directly from the pipeline\'s recorded '
     'selection log. The binary outcome collapsing rule is stated in Section 3.1.')
para(doc,
     'On the absence of a covariate summary table, the reviewer is correct and we have added one. The '
     'new Table D3 reports the distribution of all twelve variables across the three nested '
     'cohorts — the cleaned pooled dataset (n = 707), the held-out test split (n = 142), and the '
     'recourse cohort (n = 48) — with continuous variables as mean (SD) and categorical variables as '
     'n (%) or the percentage in each level. We would draw the reviewer\'s attention to what it shows, '
     'because it answers the final sentence of the comment more usefully than a reassurance would. The '
     'test split is closely comparable to the pooled dataset on every variable, as stratified sampling '
     'would predict. The recourse cohort is not: it is older (57.1 vs 52.8 years), more predominantly '
     'male (91.7% vs 76.4%), and substantially more symptomatic — exercise-induced angina in 70.8% '
     'versus 38.2%, chest-pain type 4 in 83.3% versus 48.9%, a flat or down-sloping ST segment in 93.7% '
     'versus 52.5%, higher ST depression (1.7 vs 0.9 mm) and lower maximum heart rate (126 vs 140 bpm).')
para(doc,
     'Our reading of this is that the 48-patient cohort does represent a coherent clinical population, '
     'but a narrow and specific one: older, predominantly male, symptomatic patients with a heavy '
     'ischaemic burden. That coherence is a direct consequence of the selection rule, which restricts '
     'the cohort to patients the classifier both labels and correctly predicts as high-risk and '
     'therefore concentrates the most unambiguously diseased profiles. We have stated this in the text '
     'accompanying Table 4, together with the two consequences that follow: the recourse results '
     'describe this phenotype and should not be extrapolated to the broader CVD population, and the '
     'small female subgroup in Section 4.9 is a structural consequence of the same concentration. We '
     'consider this a more useful answer to the reviewer\'s question than a claim of representativeness '
     'would have been, since the cohort is demonstrably not representative and the manuscript now says '
     'so with the supporting distributions in view.')

# ---------------------------------------------------------------- closing
banner(doc, 'Closing Statement')
para(doc,
     'We thank the Editor and Reviewer 3 once again for the care taken over this manuscript across '
     'several rounds. The revision has, we believe, improved the paper in the direction the reviewer '
     'has consistently pushed: the claims are now bounded by what a cross-sectional design can support, '
     'the uncertainty is reported at the level that matters for interpretation rather than at the level '
     'that happens to be narrower, and the cohort is documented in enough detail for a reader to judge '
     'its scope for themselves. We have added the analyses that were missing rather than deferring '
     'them, and where we have not followed a recommendation we have given our reasoning and the '
     'underlying figures explicitly so that the Editor and reviewer can assess it.')
para(doc,
     'We recognise that Comments 1 and 2 identify limitations that this dataset cannot resolve, and we '
     'have not represented them as resolved. Our position is that the contribution — a reproducible '
     'framework for generating recourse candidates and quantifying what a structural-coherence layer '
     'contributes to them, evaluated honestly and with its assumptions stated — is a methodological one '
     'that stands within those limits, and that the manuscript now claims no more than that. We would '
     'welcome the Editor\'s guidance on whether the k-fold analysis discussed under Comment 3 should be '
     'added, and we remain willing to undertake it. We hope the revised manuscript is now suitable for '
     'publication.')

out = '/sessions/practical-gallant-noether/mnt/outputs/v8work/Response_to_Reviewers_R3_Round2.docx'
doc.save(out)
print('saved', out)
