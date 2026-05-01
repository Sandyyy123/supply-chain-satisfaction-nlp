"""
Build notebooks/03_modeling.ipynb programmatically.
Run once: python3 src/build_modeling_notebook.py
Then execute via:
    python3 -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 notebooks/03_modeling.ipynb
"""
import nbformat as nbf

cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Olist CSAT Modeling: Baseline + Improved

**Project:** Liora #2 (Supply Chain - Customer Satisfaction)

**Goal:** Binary classification of customer satisfaction (review_score >= 4 vs <= 3)
on the Olist Brazilian e-commerce dataset. Train four tabular baselines
(Logistic Regression, Random Forest, XGBoost, LightGBM) on engineered supply-chain
features, then improve with TF-IDF on Portuguese review text in a late-fusion ensemble.
Compare class weighting vs SMOTE for the imbalance (positive class ~77%).

Outputs:
- `deliverables/csat_lgbm.pkl`: trained LightGBM model + preprocessor + feature list
- `deliverables/metrics.json`: full metrics for every model

**Inputs:** 9 olist CSVs joined on `order_id` / `customer_id`. Cohort restricted to
delivered orders with a non-null delivery date and a review score (~95.8k rows).
""")

code("""
import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)

import xgboost as xgb
import lightgbm as lgb

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 160)

PROJECT = '/root/AI/liora_projects/02_supply_chain_csat'
DATA_DIR = os.path.join(PROJECT, 'data')
DELIVERABLES = os.path.join(PROJECT, 'deliverables')
os.makedirs(DELIVERABLES, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print('Versions:', pd.__version__, 'pandas /', xgb.__version__, 'xgboost /', lgb.__version__, 'lightgbm')
""")

md("## 1. Load and join the relevant tables")

code("""
orders = pd.read_csv(os.path.join(DATA_DIR, 'olist_orders_dataset.csv'))
items = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_items_dataset.csv'))
customers = pd.read_csv(os.path.join(DATA_DIR, 'olist_customers_dataset.csv'))
reviews = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_reviews_dataset.csv'))
payments = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_payments_dataset.csv'))
products = pd.read_csv(os.path.join(DATA_DIR, 'olist_products_dataset.csv'))

print({k: v.shape for k, v in dict(orders=orders, items=items, customers=customers,
                                   reviews=reviews, payments=payments, products=products).items()})
""")

code("""
# Parse timestamp columns on orders
ts_cols = ['order_purchase_timestamp', 'order_approved_at',
           'order_delivered_carrier_date', 'order_delivered_customer_date',
           'order_estimated_delivery_date']
for c in ts_cols:
    orders[c] = pd.to_datetime(orders[c], errors='coerce')

# Restrict cohort to delivered orders with valid delivery date
orders_d = orders[(orders['order_status'] == 'delivered') &
                  orders['order_delivered_customer_date'].notna()].copy()
print('delivered cohort:', orders_d.shape)
""")

code("""
# Aggregate order_items to order level
items_agg = items.groupby('order_id').agg(
    items_count=('order_item_id', 'max'),
    n_distinct_products=('product_id', 'nunique'),
    n_distinct_sellers=('seller_id', 'nunique'),
    price_sum=('price', 'sum'),
    freight_sum=('freight_value', 'sum'),
).reset_index()

# Aggregate payments
pay_agg = payments.groupby('order_id').agg(
    payment_total=('payment_value', 'sum'),
    n_payments=('payment_sequential', 'max'),
    max_installments=('payment_installments', 'max'),
).reset_index()

# Most common payment type per order
pay_type = (payments.sort_values('payment_value', ascending=False)
            .drop_duplicates('order_id')[['order_id', 'payment_type']])

# Reviews: keep one per order (latest)
reviews_sorted = reviews.sort_values('review_creation_date', ascending=False)
reviews_unique = reviews_sorted.drop_duplicates('order_id')[
    ['order_id', 'review_score', 'review_comment_message']
]

# Product category via items -> products
item_cat = (items.merge(products[['product_id', 'product_category_name']],
                        on='product_id', how='left')
            .groupby('order_id')['product_category_name']
            .agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else np.nan)
            .reset_index()
            .rename(columns={'product_category_name': 'top_category'}))

print('aggregations built')
""")

code("""
# Final join
df = (orders_d
      .merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')
      .merge(items_agg, on='order_id', how='left')
      .merge(pay_agg, on='order_id', how='left')
      .merge(pay_type, on='order_id', how='left')
      .merge(item_cat, on='order_id', how='left')
      .merge(reviews_unique, on='order_id', how='inner'))

print('joined shape:', df.shape)
print('review_score coverage:', df['review_score'].notna().mean())
""")

md("## 2. Feature engineering")

code("""
# Time-based features
df['delivery_days'] = (df['order_delivered_customer_date']
                       - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0
df['estimated_days'] = (df['order_estimated_delivery_date']
                        - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0
df['delivery_delay'] = df['delivery_days'] - df['estimated_days']
df['late_flag'] = (df['delivery_delay'] > 0).astype(int)

df['approval_lag_h'] = (df['order_approved_at']
                        - df['order_purchase_timestamp']).dt.total_seconds() / 3600.0
df['carrier_lag_d'] = (df['order_delivered_carrier_date']
                       - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0

# Calendar features
df['purchase_weekday'] = df['order_purchase_timestamp'].dt.weekday
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

# Basket / freight features
df['freight_ratio'] = df['freight_sum'] / df['price_sum'].replace(0, np.nan)
df['avg_item_price'] = df['price_sum'] / df['items_count'].replace(0, np.nan)
df['freight_per_item'] = df['freight_sum'] / df['items_count'].replace(0, np.nan)

# Comment text feature: presence + length
df['has_comment'] = df['review_comment_message'].notna().astype(int)
df['comment_len'] = df['review_comment_message'].fillna('').str.len()

# Binary target: satisfied = review_score >= 4
df['target'] = (df['review_score'] >= 4).astype(int)

print('positive rate:', df['target'].mean().round(4))
print('shape:', df.shape)
""")

code("""
# Reduce high-cardinality categoricals
top_states = df['customer_state'].value_counts().nlargest(10).index
df['state_top'] = df['customer_state'].where(df['customer_state'].isin(top_states), 'OTHER')

top_cats = df['top_category'].value_counts().nlargest(15).index
df['cat_top'] = df['top_category'].where(df['top_category'].isin(top_cats), 'OTHER').fillna('UNK')

df['pay_type'] = df['payment_type'].fillna('UNK')

# Tabular feature lists
num_features = [
    'delivery_days', 'estimated_days', 'delivery_delay', 'late_flag',
    'approval_lag_h', 'carrier_lag_d',
    'purchase_weekday', 'purchase_month', 'purchase_hour',
    'items_count', 'n_distinct_products', 'n_distinct_sellers',
    'price_sum', 'freight_sum', 'freight_ratio', 'avg_item_price', 'freight_per_item',
    'payment_total', 'n_payments', 'max_installments',
    'has_comment', 'comment_len',
]
cat_features = ['state_top', 'cat_top', 'pay_type']

# Fill numeric NaNs with column median for tree models, scaler will handle LR
for c in num_features:
    df[c] = pd.to_numeric(df[c], errors='coerce')
print('feature counts:', len(num_features), 'numeric,', len(cat_features), 'categorical')
""")

md("""
**Feature engineering rationale**

- `delivery_days` and `delivery_delay` are the strongest single predictors per the EDA
  (Spearman -0.235 and -0.176). 1-star orders are 12x more likely to be late.
- `late_flag` captures the threshold effect: customers seem to react sharply when the
  promise is missed, not just to absolute delivery time.
- `freight_ratio` (freight / price) proxies "I paid a lot to ship a cheap thing",
  which correlates with low scores in low-value bulky categories.
- `items_count`, `n_distinct_sellers` capture basket complexity; multi-seller orders
  have multiple shipping legs, more failure points.
- `purchase_weekday/month/hour` catch seasonality (year-end peak, weekend orders).
- `has_comment` and `comment_len` are leakage-safe meta-text features: angry customers
  tend to write longer comments. The actual text goes into the TF-IDF arm later.
- Top-10 states and top-15 categories collapse long tails to keep one-hot dim sane.
""")

md("## 3. Train / val / test split")

code("""
# Stratified 70/15/15 split
df_model = df[num_features + cat_features + ['review_comment_message', 'target']].dropna(
    subset=['delivery_days', 'estimated_days']
).reset_index(drop=True)

print('modeling rows:', len(df_model))

X_temp, X_test, y_temp, y_test = train_test_split(
    df_model.drop(columns=['target']), df_model['target'],
    test_size=0.15, stratify=df_model['target'], random_state=RANDOM_STATE)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE)

print('train:', X_train.shape, 'val:', X_val.shape, 'test:', X_test.shape)
print('train pos rate:', y_train.mean().round(4))
""")

code("""
# Build preprocessor for tabular models
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(with_mean=True), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
    ],
    remainder='drop',
    verbose_feature_names_out=False,
)

# Fit on train, transform all splits
X_train_t = preprocessor.fit_transform(X_train.copy().fillna({c: X_train[c].median()
                                                              for c in num_features}))
X_val_t   = preprocessor.transform(X_val.copy().fillna({c: X_train[c].median()
                                                        for c in num_features}))
X_test_t  = preprocessor.transform(X_test.copy().fillna({c: X_train[c].median()
                                                         for c in num_features}))

feat_names = preprocessor.get_feature_names_out()
print('transformed train shape:', X_train_t.shape, 'features:', len(feat_names))
""")

md("## 4. Baseline tabular models")

code("""
def evaluate(name, y_true, y_pred, y_proba):
    return {
        'model': name,
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'pr_auc': float(average_precision_score(y_true, y_proba)),
        'f1_pos': float(f1_score(y_true, y_pred, pos_label=1)),
        'f1_neg': float(f1_score(y_true, y_pred, pos_label=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
    }

results = []

# 4.1 Logistic Regression with class weighting
t0 = time.time()
lr = LogisticRegression(max_iter=2000, class_weight='balanced',
                        n_jobs=-1, random_state=RANDOM_STATE)
lr.fit(X_train_t, y_train)
proba = lr.predict_proba(X_val_t)[:, 1]
pred = (proba >= 0.5).astype(int)
r = evaluate('logreg_balanced', y_val, pred, proba)
r['fit_seconds'] = round(time.time() - t0, 1)
results.append(r); print(r)
""")

code("""
# 4.2 Random Forest
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=10,
                            class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(X_train_t, y_train)
proba = rf.predict_proba(X_val_t)[:, 1]
pred = (proba >= 0.5).astype(int)
r = evaluate('random_forest', y_val, pred, proba)
r['fit_seconds'] = round(time.time() - t0, 1)
results.append(r); print(r)
""")

code("""
# 4.3 XGBoost
t0 = time.time()
neg_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, scale_pos_weight=1.0,
    tree_method='hist', eval_metric='auc',
    n_jobs=-1, random_state=RANDOM_STATE,
)
xgb_clf.fit(X_train_t, y_train,
            eval_set=[(X_val_t, y_val)], verbose=False)
proba = xgb_clf.predict_proba(X_val_t)[:, 1]
pred = (proba >= 0.5).astype(int)
r = evaluate('xgboost', y_val, pred, proba)
r['fit_seconds'] = round(time.time() - t0, 1)
results.append(r); print(r)
""")

code("""
# 4.4 LightGBM
t0 = time.time()
lgb_clf = lgb.LGBMClassifier(
    n_estimators=600, num_leaves=63, learning_rate=0.05,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    class_weight='balanced',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
)
lgb_clf.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
proba = lgb_clf.predict_proba(X_val_t)[:, 1]
pred = (proba >= 0.5).astype(int)
r = evaluate('lightgbm_balanced', y_val, pred, proba)
r['fit_seconds'] = round(time.time() - t0, 1)
results.append(r); print(r)
""")

code("""
baseline_table = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
baseline_table
""")

md("## 5. Confusion matrix and ROC curve for best baseline")

code("""
best_baseline_name = baseline_table.iloc[0]['model']
print('best baseline (val):', best_baseline_name)

best_model = {
    'logreg_balanced': lr,
    'random_forest': rf,
    'xgboost': xgb_clf,
    'lightgbm_balanced': lgb_clf,
}[best_baseline_name]

proba_val = best_model.predict_proba(X_val_t)[:, 1]
pred_val = (proba_val >= 0.5).astype(int)

cm = confusion_matrix(y_val, pred_val)
print('Confusion matrix (val):')
print(pd.DataFrame(cm, index=['true_neg', 'true_pos'], columns=['pred_neg', 'pred_pos']))
print(classification_report(y_val, pred_val, digits=3))
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# ROC
fpr, tpr, _ = roc_curve(y_val, proba_val)
axes[0].plot(fpr, tpr, lw=2, label=f'{best_baseline_name} (AUC={roc_auc_score(y_val, proba_val):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('ROC (val)')
axes[0].legend(loc='lower right')

# Confusion matrix heatmap
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['pred neg', 'pred pos'])
axes[1].set_yticklabels(['true neg', 'true pos'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
axes[1].set_title('Confusion matrix (val)')
plt.tight_layout()
plt.show()
""")

md("## 6. Top features (best baseline)")

code("""
def feature_importance_df(model, names):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_[0])
    else:
        return None
    return (pd.DataFrame({'feature': names, 'importance': imp})
            .sort_values('importance', ascending=False)
            .reset_index(drop=True))

fi = feature_importance_df(best_model, feat_names)
top20 = fi.head(20)
top20
""")

code("""
# Save baseline metrics so far
baseline_results = {
    'best_baseline': best_baseline_name,
    'val_metrics': results,
    'top20_features': top20.to_dict(orient='records'),
}
""")

md("""
## 7. Improved model: TF-IDF on Portuguese review text + late fusion

We add a TF-IDF representation of `review_comment_message` (Portuguese, 41k reviews
have text). Two paths:

1. **Late fusion**: train a separate text-only LightGBM on TF-IDF, then average its
   probability with the tabular best-baseline probability.
2. **Stacked sparse**: hstack TF-IDF onto the tabular feature matrix and train a
   single LightGBM on the combined sparse matrix.

We try both and pick the better one on validation. We also compare class-weighting
vs SMOTE on the tabular features alone.
""")

code("""
# 7.1 Build TF-IDF on training comments only (rows with comment present)
train_text = X_train['review_comment_message'].fillna('').astype(str)
val_text   = X_val['review_comment_message'].fillna('').astype(str)
test_text  = X_test['review_comment_message'].fillna('').astype(str)

# Portuguese stopword-ish: keep simple (no stopword list dependency); rely on min_df
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    strip_accents='unicode',
    lowercase=True,
    sublinear_tf=True,
)
Xtr_txt = tfidf.fit_transform(train_text)
Xva_txt = tfidf.transform(val_text)
Xte_txt = tfidf.transform(test_text)
print('tfidf train shape:', Xtr_txt.shape, 'nnz/row:', Xtr_txt.nnz / Xtr_txt.shape[0])
""")

code("""
# 7.2 Text-only LightGBM on TF-IDF
t0 = time.time()
lgb_txt = lgb.LGBMClassifier(
    n_estimators=400, num_leaves=63, learning_rate=0.05,
    feature_fraction=0.6, bagging_fraction=0.9, bagging_freq=5,
    class_weight='balanced',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
)
lgb_txt.fit(Xtr_txt, y_train, eval_set=[(Xva_txt, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
proba_txt_val = lgb_txt.predict_proba(Xva_txt)[:, 1]
r_txt = evaluate('lgb_text_only', y_val, (proba_txt_val >= 0.5).astype(int), proba_txt_val)
r_txt['fit_seconds'] = round(time.time() - t0, 1)
print(r_txt)
""")

code("""
# 7.3 Stacked sparse: tabular + TF-IDF
Xtr_stack = sparse.hstack([sparse.csr_matrix(X_train_t), Xtr_txt]).tocsr()
Xva_stack = sparse.hstack([sparse.csr_matrix(X_val_t),   Xva_txt]).tocsr()
Xte_stack = sparse.hstack([sparse.csr_matrix(X_test_t),  Xte_txt]).tocsr()

t0 = time.time()
lgb_stack = lgb.LGBMClassifier(
    n_estimators=800, num_leaves=63, learning_rate=0.05,
    feature_fraction=0.6, bagging_fraction=0.9, bagging_freq=5,
    class_weight='balanced',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
)
lgb_stack.fit(Xtr_stack, y_train, eval_set=[(Xva_stack, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
proba_stack_val = lgb_stack.predict_proba(Xva_stack)[:, 1]
r_stack = evaluate('lgb_stacked_tfidf', y_val,
                   (proba_stack_val >= 0.5).astype(int), proba_stack_val)
r_stack['fit_seconds'] = round(time.time() - t0, 1)
print(r_stack)
""")

code("""
# 7.4 Late fusion: average tabular best-baseline proba with text proba
proba_tab_val = best_model.predict_proba(X_val_t)[:, 1]
proba_fuse_val = 0.5 * proba_tab_val + 0.5 * proba_txt_val
r_fuse = evaluate('late_fusion_tab_text', y_val,
                  (proba_fuse_val >= 0.5).astype(int), proba_fuse_val)
print(r_fuse)
""")

md("### 7.5 Class weighting vs SMOTE comparison")

code("""
# Tabular only, LightGBM, no class_weight, with SMOTE oversampling on train
t0 = time.time()
sm = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
Xtr_sm, ytr_sm = sm.fit_resample(X_train_t, y_train)
print('SMOTE-resampled train:', Xtr_sm.shape, 'pos rate:', ytr_sm.mean().round(3))

lgb_sm = lgb.LGBMClassifier(
    n_estimators=600, num_leaves=63, learning_rate=0.05,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
)
lgb_sm.fit(Xtr_sm, ytr_sm, eval_set=[(X_val_t, y_val)],
           callbacks=[lgb.early_stopping(30, verbose=False)])
proba_sm_val = lgb_sm.predict_proba(X_val_t)[:, 1]
r_sm = evaluate('lgb_smote_tabular', y_val,
                (proba_sm_val >= 0.5).astype(int), proba_sm_val)
r_sm['fit_seconds'] = round(time.time() - t0, 1)
print(r_sm)
""")

code("""
# Roll up the improved-arm comparison
improved_results = [r_txt, r_stack, r_fuse, r_sm]
improved_table = pd.DataFrame(improved_results).sort_values('roc_auc', ascending=False)
improved_table
""")

md("## 8. Final test-set evaluation: best baseline vs best improved")

code("""
# Pick the best model on val ROC-AUC across baseline + improved arms
all_val = pd.DataFrame(results + improved_results).sort_values('roc_auc', ascending=False)
print('full val leaderboard:')
print(all_val[['model', 'roc_auc', 'pr_auc', 'f1_pos', 'balanced_accuracy']])
""")

code("""
# Score the chosen final model on the held-out test set
def score_on_test(name):
    if name == 'logreg_balanced':
        proba = lr.predict_proba(X_test_t)[:, 1]
    elif name == 'random_forest':
        proba = rf.predict_proba(X_test_t)[:, 1]
    elif name == 'xgboost':
        proba = xgb_clf.predict_proba(X_test_t)[:, 1]
    elif name == 'lightgbm_balanced':
        proba = lgb_clf.predict_proba(X_test_t)[:, 1]
    elif name == 'lgb_text_only':
        proba = lgb_txt.predict_proba(Xte_txt)[:, 1]
    elif name == 'lgb_stacked_tfidf':
        proba = lgb_stack.predict_proba(Xte_stack)[:, 1]
    elif name == 'late_fusion_tab_text':
        pt = best_model.predict_proba(X_test_t)[:, 1]
        px = lgb_txt.predict_proba(Xte_txt)[:, 1]
        proba = 0.5 * pt + 0.5 * px
    elif name == 'lgb_smote_tabular':
        proba = lgb_sm.predict_proba(X_test_t)[:, 1]
    else:
        raise ValueError(name)
    pred = (proba >= 0.5).astype(int)
    r = evaluate(name + '_TEST', y_test, pred, proba)
    return r, proba

baseline_test, _ = score_on_test(best_baseline_name)
print('Baseline test:', baseline_test)

best_improved_name = all_val.iloc[0]['model']
improved_test, improved_test_proba = score_on_test(best_improved_name)
print('Improved test (' + best_improved_name + '):', improved_test)
""")

code("""
# Confusion matrix + ROC for the final improved model on test
pred_test = (improved_test_proba >= 0.5).astype(int)
cm_test = confusion_matrix(y_test, pred_test)
print('Test confusion matrix:')
print(pd.DataFrame(cm_test, index=['true_neg', 'true_pos'], columns=['pred_neg', 'pred_pos']))
print(classification_report(y_test, pred_test, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fpr, tpr, _ = roc_curve(y_test, improved_test_proba)
axes[0].plot(fpr, tpr, lw=2, label=f'{best_improved_name} (AUC={improved_test["roc_auc"]:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('ROC (test)')
axes[0].legend(loc='lower right')

axes[1].imshow(cm_test, cmap='Blues')
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['pred neg', 'pred pos'])
axes[1].set_yticklabels(['true neg', 'true pos'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm_test[i, j], ha='center', va='center',
                     color='white' if cm_test[i, j] > cm_test.max()/2 else 'black')
axes[1].set_title('Confusion matrix (test)')
plt.tight_layout()
plt.show()
""")

md("## 9. Save deliverables")

code("""
# Save the LightGBM tabular-balanced model + preprocessor + features
artifact = {
    'model': lgb_clf,
    'preprocessor': preprocessor,
    'num_features': num_features,
    'cat_features': cat_features,
    'random_state': RANDOM_STATE,
    'note': 'LightGBM tabular baseline with class_weight=balanced. '
            'Best improved model = ' + best_improved_name + ' (see metrics.json).',
}
with open(os.path.join(DELIVERABLES, 'csat_lgbm.pkl'), 'wb') as f:
    pickle.dump(artifact, f)
print('saved csat_lgbm.pkl')
""")

code("""
# Save metrics JSON
metrics = {
    'cohort_size': int(len(df_model)),
    'positive_rate_train': float(y_train.mean()),
    'positive_rate_test': float(y_test.mean()),
    'split': {'train': int(len(X_train)), 'val': int(len(X_val)), 'test': int(len(X_test))},
    'baseline_val_metrics': results,
    'improved_val_metrics': improved_results,
    'final_baseline_test': baseline_test,
    'final_improved_test': improved_test,
    'best_baseline_name': best_baseline_name,
    'best_improved_name': best_improved_name,
    'top20_features_baseline': top20.to_dict(orient='records'),
}
with open(os.path.join(DELIVERABLES, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2, default=float)
print('saved metrics.json')

print()
print('=== FINAL SUMMARY ===')
print('Baseline   (' + best_baseline_name + ') TEST: ROC-AUC =', round(baseline_test['roc_auc'], 4),
      '/ PR-AUC =', round(baseline_test['pr_auc'], 4))
print('Improved   (' + best_improved_name + ') TEST: ROC-AUC =', round(improved_test['roc_auc'], 4),
      '/ PR-AUC =', round(improved_test['pr_auc'], 4))
""")


nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}

OUT = '/root/AI/liora_projects/02_supply_chain_csat/notebooks/03_modeling.ipynb'
with open(OUT, 'w') as f:
    nbf.write(nb, f)
print('wrote', OUT, 'with', len(cells), 'cells')
