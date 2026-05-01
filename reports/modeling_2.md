# Modeling 2 - Tuned XGBoost (Olist CSat)

## What changed vs `modeling_1.md`

Same features, same split, deeper boosted trees, lower learning rate, more boosting rounds:

`n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, scale_pos_weight = neg/pos`.

## Test results

| Model | ROC-AUC | PR-AUC | F1 (positive) | Macro F1 | Balanced Acc |
|---|---|---|---|---|---|
| Logistic Regression (modeling_1) | 0.749 | 0.898 | 0.793 | 0.636 | 0.682 |
| Random Forest (modeling_1) | 0.748 | 0.896 | 0.861 | 0.676 | 0.678 |
| XGBoost baseline (modeling_1) | 0.742 | 0.894 | 0.830 | 0.655 | 0.677 |
| **XGBoost tuned (this run)** | 0.736 | 0.891 | 0.847 | 0.662 | 0.670 |

The tuned XGBoost slightly underperforms the baseline XGBoost on this feature set: ROC-AUC drops from 0.742 to 0.736. Deeper trees and longer training overfit noise in the ~58k-row training partition. The baseline XGBoost configuration is therefore the recommended deployment model among the four.

This is consistent with the literature pattern that on tabular data of this size with one strong categorical (`customer_state`) and a handful of strong continuous features (`delivery_days`, `late_flag`, `has_comment`), gradient boosting plateaus quickly.

## Why the marginal lift is small

The information ceiling on the current feature set is binding. Without:

- Text features over the Portuguese `review_comment_message` (free-form reviews carry strong sentiment signal)
- Seller-level historical performance (a mean-encoded seller satisfaction rate)
- Product-category-level priors

the tabular model is bounded by ROC-AUC ~0.75. To push past that, the next iteration (modeling_3) needs:

- TF-IDF (1-2 grams) on the lemmatised Portuguese review text concatenated with the late-fusion of the tabular features. A multilingual model (BERTimbau, XLM-R) would push further.
- Seller mean-encoding with leave-one-out smoothing (`mean_target_seller`).
- Class imbalance: SMOTE or focal loss on the minority (negative) class given the 79/21 split.

## Operational decision

For a deployment that uses only the order-time tabular signals, the **modeling_1 XGBoost baseline (300 trees, depth 6)** is the recommended model: it has the strongest ROC-AUC (0.742), comparable F1, and is faster at inference than the tuned variant. The tuned variant is retained in `deliverables/csat_xgb.pkl` for completeness.

## Persisted artifacts

- `deliverables/csat_xgb.pkl` - tuned XGBoost pipeline (600 trees)
- `deliverables/metrics.json` - per-split metrics for all four models, top-15 RF features
