# CHANGES - Project #02 Olist Supply Chain CSat (manuscript prose reconciliation)

Date: 2026-05-08
Scope: minimal prose edits to `manuscripts/manuscript.md` only. No code, notebook, or deliverable was modified.

Trigger: Validator and Improver reports flagged two manuscript-vs-code drifts:
- ADASYN [8] is named in the Methods exclusion statement but never implemented in `src/build_modeling_notebook.py` or `notebooks/03_modeling.ipynb`.
- `has_comment` carries label-adjacent leakage that inflates ROC-AUC and Random Forest impurity importance; the prose under-stated it as "behavioural rather than content channel".

Word count: 4,236 (before) -> 4,420 (after). Inside the 4,000-5,000 target.
Em-dash count: 0 (before) -> 0 (after).

---

## Edit 1. Section 3.4 Models (final paragraph) - remove ADASYN-as-implemented claim

**Before:**

> We deliberately exclude SMOTE [7] and ADASYN [8] from the primary comparison: the dataset is large enough that class weighting provides similar benefit without injecting synthetic minority points, which is desirable when the minority class corresponds to real, semantically meaningful negative reviews rather than to noise.

**After:**

> We deliberately exclude SMOTE [7] from the primary comparison: the dataset is large enough that class weighting provides similar benefit without injecting synthetic minority points, which is desirable when the minority class corresponds to real, semantically meaningful negative reviews rather than to noise. ADASYN [8] is cited as related literature on minority oversampling but is not part of the primary comparison and is not implemented in the present pipeline; alternative oversampling strategies including ADASYN are deferred to the Phase 2 ablation outlined in section 5.4.

**Why:** Source code references SMOTE only. ADASYN now reads as "related literature, not run", which matches `src/build_modeling_notebook.py` exactly. Forward-pointing reference to section 5.4 keeps the citation [8] anchored in the bibliography without claiming the experiment was executed.

---

## Edit 2. Section 4.2 Feature importance (final paragraph) - acknowledge `has_comment` leakage plainly

**Before:**

> The high importance of `has_comment` reflects a behavioural rather than content channel: customers who write a comment are skewed toward stronger feelings in either direction, and at the time of model scoring the binary flag for whether a comment exists is a legitimate input. Adding the comment text itself as features would risk label leakage if the comment was written after the star, which is the typical pattern; this question is treated explicitly in the next modelling iteration.

**After:**

> The high importance of `has_comment` is partly behavioural (customers who write a comment are skewed toward stronger feelings in either direction) but also carries a label-adjacent component: in the Olist workflow the comment field is recorded together with the review score, so the decision to comment is not strictly anterior to the rating and the binary flag therefore encodes a fraction of the target signal it is being used to predict. The reported feature ranking and the deployed ROC-AUC are inflated to an unquantified degree by this leakage. We retain the feature in the present comparison for continuity with the EDA, flag the issue plainly here, and schedule a Phase 2 ablation (section 5.4) that drops `has_comment` and reports the ROC-AUC drop alongside a leakage-free `comment_propensity` proxy fit on tabular features only. Adding the comment text itself as features would compound the leakage risk because the comment is typically written after the star is assigned; this question is treated explicitly in the next modelling iteration.

**Why:** The previous paragraph framed `has_comment` as a "legitimate input" with no leakage acknowledgement at the point where its impurity rank is reported. The improver flagged this as understated. The new prose names the leakage explicitly, states that the reported numbers are inflated by an unquantified amount, and points to the Phase 2 ablation as the place where the corrected number will be reported.

---

## Edit 3. Section 5.3 Limitations (second item) - upgrade `has_comment` from "boundary" to "known leakage"

**Before:**

> Second, `has_comment` is at the boundary of legitimate signal and label leakage. We treat it here as a behavioural input that is observed at scoring time, but the comment text itself is intentionally excluded from the present feature set.

**After:**

> Second, `has_comment` is a known label-adjacent feature: in the Olist data model the comment is captured in the same review event as the star, so the binary "did the customer comment" flag is not strictly an order-time signal and almost certainly inflates the reported ROC-AUC. The exact magnitude of the inflation has not been measured in the present pipeline; a Phase 2 ablation that drops `has_comment` and replaces it with a tabular-only `comment_propensity` proxy is scheduled in section 5.4, and the ablation result should be treated as the headline number in any operational deployment of this model.

**Why:** Limitations section now matches the strength of the acknowledgement in 4.2 and the improver's framing of the issue (HIGH priority). The Phase 2 ablation flag is now mentioned in three places (3.4, 4.2, 5.3) and consistently points to section 5.4, which already lists the ablation under future modelling work.

---

## Sections NOT edited (and why)

- Section 5.4 (Plan for next modelling iteration): unchanged. SMOTE [7], ADASYN [8] and focal loss [10] are listed there as future-work alternatives, which is consistent with reality (none have been run yet) and is the correct location for the deferred-experiment list.
- Tables 1 and 2: unchanged. Numbers continue to come from `deliverables/metrics.json` and the Random Forest impurity ranking. Edit 2 contextualises Table 2 without modifying the numbers themselves.
- Abstract, Introduction, sections 1, 2, 3.1-3.3, 3.5, 4.1, 4.3, 5.1, 5.2, 6, References: unchanged. No drift was reported in those sections.
- All code, notebooks, and deliverables: untouched per task scope.

## Unresolved drift / known gaps after this reconciliation

- The actual ablation (drop `has_comment`, refit four models, report new ROC-AUC) has not been run. The manuscript now flags the issue and schedules the experiment, but the corrected headline number is still pending. This is an acknowledged future-work item, not a manuscript-vs-code drift.
- Section 5.4 still lists ADASYN [8] as a Phase 2 oversampling alternative; this is forward-looking and matches code (not run yet), so it does not constitute drift.
- Validator's other open items (missing `checkpoint.json`, `model_baseline.py` / `model_advanced.py` filenames, EDA notebook capitalisation) are layout deviations and are out of scope for prose-only reconciliation.
