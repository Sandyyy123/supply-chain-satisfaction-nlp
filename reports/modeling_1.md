# Modeling 1 - Baseline (Olist Customer Satisfaction Binary Classification)

## Task

Binary classification: did the order receive a review_score of 4 or 5 (positive) versus 1, 2, or 3 (non-positive). Source: Olist Brazilian e-commerce open dataset (Kaggle), pulled from the `olist_*.csv` star schema.

## Tables joined

- `olist_orders_dataset` (n=99,441) - order metadata, status, timestamps
- `olist_order_items_dataset` (n=112,650) - aggregated to one row per order (item count, price sum, freight sum, unique products / sellers)
- `olist_order_reviews_dataset` (n=99,224) - review_score, comment text presence
- `olist_customers_dataset` (n=99,441) - state for `customer_state` categorical

After filtering to `order_status == 'delivered'` and dropping rows with missing delivery date or score, the modeling table holds **96,353 orders**. Positive rate (target=1, score >= 4) = **78.9%**.

## Engineered features

13 numeric + 1 categorical:

- `delivery_days` (purchase to delivery)
- `estimated_days` (purchase to estimated delivery)
- `late_flag` (delivered after estimate)
- `weekday_purchase`, `hour_purchase`, `month_purchase`
- `items_count`, `items_price_sum`, `items_freight_sum`, `freight_ratio`
- `n_unique_products`, `n_unique_sellers`
- `has_comment` (review left a written message)
- `customer_state` (27 Brazilian states)

## Split

Stratified 60/20/20 with `random_state=42`:

- Train: 57,811
- Val: 19,271
- Test: 19,271

## Baselines (test set)

| Model | ROC-AUC | PR-AUC | F1 (positive) | Macro F1 | Balanced Acc |
|---|---|---|---|---|---|
| Logistic Regression (class_weight='balanced') | 0.749 | 0.898 | 0.793 | 0.636 | 0.682 |
| Random Forest (200 trees, depth 15, balanced) | 0.748 | 0.896 | 0.861 | 0.676 | 0.678 |
| XGBoost baseline (300 trees, depth 6, scale_pos_weight) | 0.742 | 0.894 | 0.830 | 0.655 | 0.677 |

All three baselines cluster near ROC-AUC 0.74. The class-weighted Logistic Regression has the best balanced accuracy (0.682), Random Forest the best F1 on the positive class (0.861) by being more permissive about positive predictions.

## Top features (Random Forest, impurity-based)

| Feature | Importance |
|---|---|
| delivery_days | 0.228 |
| has_comment | 0.182 |
| late_flag | 0.149 |
| estimated_days | 0.068 |
| items_freight_sum | 0.063 |
| freight_ratio | 0.056 |
| items_price_sum | 0.054 |
| hour_purchase | 0.037 |
| month_purchase | 0.035 |
| items_count | 0.035 |

Delivery-related signals dominate: actual delivery days, whether the order was late, and the presence of a written comment together account for ~56% of impurity gain. This matches the EDA finding that 1-star reviews have a 12.6x higher late-delivery rate than 5-star reviews and that mean delivery time is roughly twice as long for 1-star vs 5-star.

`has_comment` is a strong predictor because customers who write a comment tend to have stronger opinions (positive or negative); its appearance high in the importance list reflects label leakage at modeling time only if `review_comment_message` text features are added downstream.

## Configuration details

- Logistic Regression: `max_iter=400`, `class_weight='balanced'`, scaled numeric features.
- Random Forest: `n_estimators=200`, `max_depth=15`, `min_samples_leaf=3`, `class_weight='balanced'`.
- XGBoost baseline: `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.8`, `scale_pos_weight = neg/pos = 0.27`, `tree_method='hist'`.
- One-hot encoding of `customer_state`. Numeric imputation with median, categorical with most-frequent.

## Takeaways

- The three baselines are tightly clustered. ROC-AUC plateaus near 0.75 with these tabular features alone.
- Delivery time + lateness + comment presence carry most of the signal; price and basket composition contribute less.
- `modeling_2.md` documents the tuned XGBoost; the bigger lift will come from adding TF-IDF features over the Portuguese `review_comment_message` (modeling 3 / future).
