# Improvements - Project #02 Supply Chain Customer Satisfaction (Olist)

Role: IMPROVER (independent reviewer). Read-only assessment of brief.pdf, notebooks/01_eda.ipynb, notebooks/03_modeling.ipynb, src/build_modeling_notebook.py, reports/{exploration_1,modeling_1,modeling_2}.md, manuscripts/manuscript.md, manuscripts/references.md, deliverables/{metrics.json,presentation.html,csat_xgb.pkl}.

---

## Top recommendation (single highest-leverage change)

**Build the text-augmented late-fusion model that the manuscript already promises ("modeling_3"), but ship it as the headline result rather than as future work.** The current deliverable plateaus at ROC-AUC 0.749 because it ignores the 40,977 Portuguese `review_comment_message` rows that the EDA itself flags as the most informative remaining signal. A concrete, low-risk path:

1. Train a TF-IDF (1-2 grams, lemmatised, Portuguese stoplist via spaCy `pt_core_news_sm`) + SGDClassifier on `review_comment_message` only as a text-only baseline. This alone typically lifts ROC-AUC by 5-8 points on Brazilian Portuguese e-commerce reviews (cf. Souza & Filho, arXiv:2112.05459).
2. Stack the text branch and the existing XGBoost tabular branch via logistic-regression meta-learner trained on out-of-fold predictions (5-fold). Keep the comment-absent rows by passing the tabular probability through unchanged when `has_comment=0`.
3. Replace `has_comment` with a no-leakage proxy (e.g. `comment_likely` from a calibrated logistic on tabular features only) so the binary "did they comment" signal is preserved without the post-hoc shortcut.

Expected lift: ROC-AUC 0.80-0.83 based on the precedents cited in the manuscript itself. **Priority: HIGH.** This is the difference between a saturated tabular study and a publishable supply-chain + NLP fusion study, and it is already half-scoped in section 5.4 of the manuscript.

---

## Weaknesses and proposed improvements

### 1. `has_comment` is label-adjacent and inflates apparent feature importance [HIGH]

`has_comment` is the #2 feature by Random Forest impurity gain (0.182). The manuscript acknowledges the leakage risk in section 4.2 and 5.3 but keeps the feature in the deployed model. In production, a customer's choice to write a comment is at least partially driven by their (already-formed) star intent, which is exactly what we are trying to predict. The Spearman correlation between `has_comment` and a 1-3 star is published-strong on Brazilian e-commerce data.

**Action:** Run an ablation that drops `has_comment` and re-reports all four classifiers' ROC-AUC. If the drop is large (e.g. 0.749 -> 0.71), state plainly in the manuscript that the deployed feature set carries a behavioural-leakage component and report both numbers (with-leak and clean). Add a `comment_propensity` feature instead, fit on tabular features only via a held-out fold, to retain the predictive power without the circularity. The manuscript currently sells the leakage as "behavioural rather than content channel" which understates the issue.

### 2. No ordinal/regression treatment of the 1-5 score; binarisation discards 80% of the score variance [HIGH]

Section 5.3 lists this as a limitation and stops there. A 5-class ordinal target carries roughly 2.32 bits of information versus 0.74 bits for the binary target; the project is throwing away ~68% of the label entropy.

**Action:** Add a third arm to the modelling table: ordinal logistic regression (cumulative-link) and an XGBoost regressor trained on the integer score with quadratic-weighted-kappa (QWK) as the eval metric. Report QWK and Spearman rank correlation alongside ROC-AUC. This is a 1-2 day addition that strengthens the methodology section materially and gives the operations team a graded score (1-5) to act on instead of a binary positive/negative flag.

### 3. No SHAP / feature-attribution shipped despite the manuscript citing TreeSHAP [Lundberg 2017, 2020] [MEDIUM]

References 22 and 23 are in the bibliography and TreeSHAP is named in section 5.4 as the interpretability method, but no SHAP plots, no `summary_plot`, no `dependence_plot` for `delivery_days` are in deliverables/, the notebook, or the presentation. The Random Forest impurity-based importance that ships in Table 2 is known to be biased toward high-cardinality numeric features (Strobl 2007), which is exactly the regime here.

**Action:** Generate `shap.summary_plot` and a `shap.dependence_plot(delivery_days, interaction_index='late_flag')` from the deployed XGBoost baseline (`deliverables/csat_xgb.pkl` is already loadable), save as PNG to `deliverables/`, embed in `presentation.html`, and replace Table 2 in the manuscript with the SHAP global-importance ranking. The narrative about delivery dominating gets stronger when shown in SHAP space because it does not have the cardinality bias.

### 4. No calibration analysis; probabilities are reported but never reliability-checked [MEDIUM]

The manuscript reports ROC-AUC, PR-AUC, F1 and balanced accuracy. None of these speak to whether the predicted probabilities are well-calibrated, which is what an operations team actually consumes when triaging dissatisfied customers (e.g. "auto-escalate orders with predicted P(neg) > 0.6"). XGBoost with `scale_pos_weight` is known to produce miscalibrated probabilities (Niculescu-Mizil & Caruana 2005).

**Action:** Add a reliability diagram (sklearn `calibration_curve` with 10 bins) and Brier score to `metrics.json` for all four models. If the deployed XGBoost is badly miscalibrated, fit a sigmoid (Platt) or isotonic calibrator on the validation split and report post-calibration Brier. Mention the operating threshold (default 0.5 on a 78.9% positive base rate is almost certainly wrong) and pick one that maximises F1 or recall@precision-0.7 on validation.

### 5. Reproducibility gaps: no requirements.txt, no checkpoint.json, no data/README.md, single seed [MEDIUM]

`ls` of the project root shows: brief.pdf, data/, deliverables/, manuscripts/, notebooks/, reports/, src/. There is no `requirements.txt`, no `environment.yml`, no `pyproject.toml`. The QA rules call out `checkpoint.json` and `data/README.md` as expected artefacts and both are absent. The split uses `random_state=42` once; results are not averaged over seeds.

**Action:** (a) Generate `requirements.txt` via `pip freeze | grep -iE 'pandas|numpy|sklearn|scikit-learn|xgboost|lightgbm|nbformat|matplotlib|scipy|shap|spacy'`. (b) Write `data/README.md` documenting the Olist Kaggle source URL, date pulled, license (CC BY-NC-SA 4.0), schema, and SHA-256 of `olist_archive.zip`. (c) Write `checkpoint.json` with project_number=2, title, methodology, status, primary_metric=ROC-AUC, primary_metric_value=0.749. (d) Run all four models across 5 seeds (42, 0, 1, 7, 2026), report mean +/- std for ROC-AUC and PR-AUC. The current 1.3-point spread between models is plausibly within seed noise, which would change the conclusion in section 4.3.

### 6. No fairness / regional-equity audit despite the geography-driven story [MEDIUM]

Section 5.1 explicitly states that northern/northeastern Brazilian states (Bahia, RJ) get worse delivery and worse scores than São Paulo, and that this is "a geography effect mediated by logistics rather than by buyer preferences". That is a fairness statement and it is not audited. A model that uses `customer_state` as a feature will by construction predict lower satisfaction for BA/PE/AM customers, which is fine descriptively but operationally toxic if it is used to triage which customers receive proactive outreach.

**Action:** Compute per-state ROC-AUC, false-negative rate (missed dissatisfied customers), and demographic parity difference for the deployed XGBoost. If FNR is systematically higher in northern states (likely, given lower data volume), document it in section 5.3 and propose either (a) per-state recalibration, or (b) excluding state from the feature set and reporting the resulting ROC-AUC drop as the cost of fairness. This is also a credible Liora differentiator versus a stock Kaggle solution.

### 7. Time-leakage risk: 60/20/20 stratified random split ignores the 2016-2018 chronology [MEDIUM]

Olist orders span Oct 2016 to Aug 2018. The current split is stratified random; it mixes future and past orders into the same training fold. For an "operations" model this is the wrong evaluation: at deployment time the model will only see orders from before today. A model that overfits to seasonal calendar effects (`hour_purchase`, `month_purchase` are in the top 10 features) will look better on a random split than on a forward-chained split.

**Action:** Add a temporal split as a secondary evaluation: train on orders with `order_purchase_timestamp < 2018-01-01`, validate on Jan-Apr 2018, test on May-Aug 2018. Report ROC-AUC for both splits. If the temporal ROC-AUC is materially lower (e.g. 0.71 vs 0.749), drop calendar features and report the more robust number as the headline.

### 8. Presentation is title-only verified; business-action narrative not checked [LOW]

`<title>Customer Satisfaction Prediction in E-Commerce</title>` is in the HTML, but the deck has not been audited end-to-end for whether the recommendation is actionable (e.g. "renegotiate freight with carrier X for routes to BA/AM/PA where late-rate is 4x national average"). The manuscript talks to a research audience; a stakeholder audience needs a single dollar-impact slide.

**Action:** Add one slide titled "Where to act first" with a Pareto bar of estimated revenue-at-risk per state computed as (n_orders_per_state) x (P(neg) per state) x (assumed avg basket revenue R$140). Then a second slide "What the model wins us" framing the avoided-churn dollar value of correctly flagging the 20% test-set negatives. This is what converts the model from "ROC-AUC 0.749" into a proposal a supply-chain director can sign.

### 9. CatBoost listed in references but never trained [LOW]

Reference [3] is CatBoost and the manuscript section 5.2 explicitly says "CatBoost is an attractive alternative encoder because it implements ordered boosting that is robust to target leakage" for seller mean-encoding. But no CatBoost model is in the four-model table. This is a small but visible gap between cited methods and methods evaluated.

**Action:** Either drop the CatBoost mention from the discussion to match the executed work, or add a 5th row to Table 1 with CatBoost trained on the same features with `cat_features=['customer_state']` and default ordered boosting. Given the report claims an information ceiling, adding CatBoost is cheap and either confirms the ceiling (strengthens the conclusion) or breaks it (better headline result).

---

## Priority summary

- HIGH: #1 (drop has_comment leakage) | #2 (ordinal target) | Top recommendation (text fusion)
- MEDIUM: #3 (SHAP) | #4 (calibration) | #5 (reproducibility files + multi-seed) | #6 (fairness audit) | #7 (temporal split)
- LOW: #8 (business-impact slide) | #9 (CatBoost or remove citation)

## Compact summary

Output: /root/AI/liora_projects/02_supply_chain_csat/improvements.md. Top 3 findings: (1) the project plateau at ROC-AUC 0.749 is self-imposed because the manuscript-promised text fusion (BERTimbau + TF-IDF) was never executed and remains the single highest-leverage gap; (2) `has_comment` carries label-adjacent leakage that inflates the feature importance ranking and the deployed ROC-AUC and is acknowledged but not removed or quantified; (3) reproducibility infrastructure is absent (no requirements.txt, no checkpoint.json, no data/README.md, single random seed), and the random 60/20/20 split ignores the 2016-2018 chronology so the headline number is likely optimistic. No blockers. Role B complete.
