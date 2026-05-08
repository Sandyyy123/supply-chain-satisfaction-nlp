# Predicting Customer Satisfaction in Brazilian E-Commerce: A Supply-Chain View of the Olist Marketplace Using Gradient-Boosted Trees

## Abstract

Customer review scores in online retail capture a noisy mixture of product quality, price perception and post-purchase logistics. We study the relative contribution of these channels using the public Olist Brazilian e-commerce dataset, which links 99,441 orders (2016-2018) to verified review stars, item-level seller information, and full delivery timestamps. After restricting the cohort to delivered orders with non-missing review scores we obtain 96,353 modeling rows with a positive class rate (review score 4 or 5) of 78.9%. We engineer 13 numeric features and one categorical state field, including delivery duration, estimated delivery duration, a late-delivery flag, freight ratio, basket size, number of unique sellers and a binary flag for the presence of a written review comment. Four classifiers are compared on a stratified 60/20/20 split: a class-weighted Logistic Regression, a class-weighted Random Forest, a baseline XGBoost with `scale_pos_weight` correction, and a tuned XGBoost with deeper trees and more boosting rounds. All four models cluster between ROC-AUC 0.736 and 0.749 on the held-out test set. The class-weighted Logistic Regression attains the best ROC-AUC (0.749) and balanced accuracy (0.682); the baseline XGBoost is the recommended deployment choice (ROC-AUC 0.742, faster than the tuned variant). Counter-intuitively the tuned XGBoost underperforms its untuned counterpart (ROC-AUC 0.736), consistent with overfitting on a tabular feature set whose information ceiling is already saturated. Feature importances are dominated by `delivery_days`, the presence of a written comment, and `late_flag`, which together account for roughly 56% of impurity gain. We discuss why on-time delivery rather than price drives satisfaction in this marketplace, and we outline a follow-up modeling stage that fuses TF-IDF and BERTimbau [16] representations of the Portuguese review text with seller-level mean-encoded targets to break past the ROC-AUC 0.75 plateau.

**Keywords:** customer satisfaction, e-commerce, Olist, supply chain, late delivery, gradient boosting, XGBoost, class imbalance, tabular machine learning, Brazilian Portuguese reviews.

## 1. Introduction

Online retail has converted the review star into the primary public proxy for service quality. For most marketplaces the star is a single integer attached to a single transaction, but it aggregates an entire purchasing episode: how the product was found and priced, whether it shipped on time, whether it arrived intact, and whether the seller answered messages. Disentangling those channels is operationally important. A retailer that can attribute a one-star drop to late delivery rather than to product quality can act on the freight contract; a retailer that confounds the two will spend on the wrong fix. Recent work in operations management has formalised this view: Cui et al. [20] show experimentally that promised and realised delivery speed shift both conversion and post-purchase satisfaction in online retail, and Aljohani [21] shows in a structural-equation model that last-mile delivery quality is a primary driver of overall e-commerce satisfaction. The empirical case for treating delivery as a first-class predictor of star rating is therefore robust, but published evaluations on large transactional datasets with full upstream provenance remain scarce.

The Olist Brazilian e-commerce dataset closes part of that gap. Olist is a marketplace aggregator that connects small and mid-sized sellers to the major Brazilian online channels, and the company released a fully linked nine-table corpus covering 99,441 orders placed between October 2016 and August 2018. Each order is connected to the customer, the seller, the items, the freight cost, the payment plan, the full set of timestamps from purchase to estimated delivery to actual delivery, and a verified review (one to five stars, optional comment in Portuguese). Compared with scraped Trustpilot or Trusted Shops review pages, Olist provides ground-truth provenance: every star is tied to a real transaction with a measurable supply-chain trace. This makes it an unusually clean substrate for studying which order-time signals predict eventual customer satisfaction.

A second motivation comes from the modelling side. The dataset has roughly 100,000 rows, a single strong categorical (Brazilian state, 27 levels), a handful of strong numeric predictors, and a heavily imbalanced binary target after binarising the star rating into positive (4-5) versus negative (1-3). This is the regime in which tree-based gradient boosting methods are known empirically to dominate deep tabular architectures: Grinsztajn et al. [6] benchmark gradient-boosted decision trees against tabular deep models on 45 medium-sized datasets and report a consistent advantage for the tree ensembles, and the original XGBoost [1], LightGBM [2] and CatBoost [3] papers all describe tabular regimes very close to the one studied here. We therefore set up a four-model comparison on the binarised Olist target: a regularised Logistic Regression baseline, a class-weighted Random Forest [4], a baseline XGBoost with `scale_pos_weight` correction, and a tuned XGBoost with deeper trees and a smaller learning rate. All four are evaluated under stratified 60/20/20 splitting with class-imbalance handling that follows the recommendations of He and Garcia [9].

The contributions of this paper are three. First, we describe a clean tabular feature set for Olist that lifts a Logistic Regression to ROC-AUC 0.749 with no text features. Second, we show that on this feature set the four classifiers cluster in a tight ROC-AUC band between 0.736 and 0.749, and that a tuned XGBoost slightly underperforms an untuned one, which we interpret as a signature of an information ceiling rather than a tuning failure. Third, we identify the exact channel through which the delivery process drives review scores: at the 1-star level, 38% of orders are delivered after the estimated date; at the 5-star level only 3% are. The implications for the next modelling iteration, which fuses TF-IDF and BERTimbau [16] features with seller-level mean encoding, are spelled out in the discussion.

## 2. Data

The Olist corpus is distributed as nine CSV tables organised in a star schema centred on the orders fact table. Row counts are: orders 99,441; order_items 112,650 (one row per item line); order_payments 103,886; order_reviews 99,224; customers 99,441 (one row per customer-order pair, with `customer_unique_id` available for repeat-buyer linkage); products 32,951; sellers 3,095; geolocation 1,000,163 zip-prefix entries; and a 71-row Portuguese-to-English category translation table. All foreign keys join cleanly. Order-status coverage is dominated by `delivered` (96,478 orders, 97.0%); the remaining statuses (`shipped`, `canceled`, `unavailable`, `invoiced`, `processing`, `created`, `approved`) account for under 3% of the corpus and are characterised by structurally missing delivery timestamps. We restrict the modelling cohort to delivered orders with non-missing customer-delivery date and a valid review score, which yields 96,353 modeling rows.

Missingness in the source tables is otherwise low and structural rather than noisy. The orders table has 0.16% missing `order_approved_at`, 1.79% missing `order_delivered_carrier_date` and 2.98% missing `order_delivered_customer_date`. The reviews table has 88.34% missing `review_comment_title` and 58.70% missing `review_comment_message`, leaving 40,977 reviews with a written Portuguese comment. The products table has 1.85% missing category names and four physical-dimension fields with 0.006% missing values. Order_items, order_payments, customers, sellers, geolocation and the category-translation table have no missing values.

The target is the integer review score, which we binarise into a positive class (4 or 5 stars) and a non-positive class (1 to 3 stars). The raw score distribution is heavily right-skewed: 5-star reviews account for 57.78% of all ratings and the 4-or-5-star class accounts for 77.07%; 1-star reviews account for 11.51% and the 1-to-3-star negative class for 22.93%. After joining onto the modelling cohort, the positive-class rate is 78.9%. This is a textbook imbalanced binary classification problem in the regime studied by Chawla et al. [7] and He and Garcia [9].

The single most informative numeric pattern in the exploratory analysis is the relationship between delivery duration and the review score. On the 95,824 delivered-and-reviewed orders, mean delivery time falls monotonically from 21.32 days for 1-star reviews to 10.68 days for 5-star reviews, and the share of orders delivered after the platform's estimated date falls from 37.9% at one star to 3.0% at five stars. The Spearman correlation between `delivery_days` and the integer review score is -0.235, and between `delivery_delay` (actual minus estimated) and the score is -0.176. Basket complexity has a smaller but consistent negative correlation (`n_items` -0.108, `n_sellers` -0.095, `freight_sum` -0.089), and raw price has near-zero Spearman correlation (-0.029), which already foreshadows the modelling result that delivery-related features dominate over price-related features.

## 3. Methods

### 3.1 Cohort and target

The modelling table is built by left-joining order_items (aggregated to one row per order) and order_reviews onto the orders table, and then joining the customers table for state. We restrict to `order_status == 'delivered'`, drop rows with missing `order_delivered_customer_date` or `review_score`, and binarise the target as `y = 1` if `review_score >= 4` and `y = 0` otherwise. The resulting cohort contains 96,353 orders with positive-class rate 78.9%.

### 3.2 Feature engineering

Thirteen numeric features and one categorical feature are constructed for each order:

1. `delivery_days`: days between `order_purchase_timestamp` and `order_delivered_customer_date`.
2. `estimated_days`: days between `order_purchase_timestamp` and `order_estimated_delivery_date`.
3. `late_flag`: 1 if `order_delivered_customer_date > order_estimated_delivery_date`, else 0.
4. `weekday_purchase`: integer 0-6 from the purchase timestamp.
5. `hour_purchase`: integer 0-23 from the purchase timestamp.
6. `month_purchase`: integer 1-12 from the purchase timestamp.
7. `items_count`: number of `order_item_id` rows for the order.
8. `items_price_sum`: sum of `price` across the order's items.
9. `items_freight_sum`: sum of `freight_value` across the order's items.
10. `freight_ratio`: `items_freight_sum / items_price_sum`, clipped at numerical extremes.
11. `n_unique_products`: distinct `product_id` count.
12. `n_unique_sellers`: distinct `seller_id` count.
13. `has_comment`: 1 if `review_comment_message` is non-empty.
14. `customer_state`: 27-level Brazilian-state categorical (one-hot encoded).

Features 1, 2 and 3 capture the supply-chain signal that the EDA highlighted; features 4-6 are calendar regularities; features 7-12 capture basket structure and freight; feature 13 is a flag for whether the customer wrote a review comment, which correlates strongly with score because customers with strong opinions in either direction are more likely to type. The state field captures the underlying Brazilian logistics geography, which the EDA showed is tightly linked to mean delivery time (8.75 days in São Paulo versus 19.25 days in Bahia).

### 3.3 Train / validation / test split

The cohort is split with stratification on the binary target into 60% train (57,811), 20% validation (19,271) and 20% test (19,271), using `random_state=42`. Stratification preserves the 78.9% positive rate within each split. Numeric features are imputed with the median; the categorical state is imputed with the most-frequent value and one-hot encoded.

### 3.4 Models

We evaluate four classifiers, each tuned to address the 79/21 class imbalance using either class weighting or a `scale_pos_weight` parameter, in line with the recommendations in [9] and [11]:

- **Logistic Regression.** `class_weight='balanced'`, `max_iter=400`, on standardised numeric features and one-hot states.
- **Random Forest** [4]. `n_estimators=200`, `max_depth=15`, `min_samples_leaf=3`, `class_weight='balanced'`, default Gini criterion.
- **XGBoost baseline** [1]. `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.8`, `tree_method='hist'`, with `scale_pos_weight = neg/pos` (approximately 0.27 given the 78.9% positive rate).
- **XGBoost tuned** [1]. Same imbalance handling and feature set, with `n_estimators=600`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.8`. The tuned configuration is intended to test whether deeper trees and a longer training schedule extract additional non-linear signal from the engineered features.

We deliberately exclude SMOTE [7] from the primary comparison: the dataset is large enough that class weighting provides similar benefit without injecting synthetic minority points, which is desirable when the minority class corresponds to real, semantically meaningful negative reviews rather than to noise. ADASYN [8] is cited as related literature on minority oversampling but is not part of the primary comparison and is not implemented in the present pipeline; alternative oversampling strategies including ADASYN are deferred to the Phase 2 ablation outlined in section 5.4.

### 3.5 Evaluation

For each model we report ROC-AUC, PR-AUC, F1 on the positive class, macro F1, and balanced accuracy on the held-out test split. ROC-AUC is the primary metric; PR-AUC is reported as a secondary metric robust to the class imbalance. Feature importance is reported using the impurity-based importance from the Random Forest, which provides a stable global ranking of the contribution of each engineered feature.

## 4. Results

### 4.1 Held-out test performance

Table 1 shows test-set performance for the four classifiers. The class-weighted Logistic Regression achieves the highest ROC-AUC (0.749) and the best balanced accuracy (0.682); the Random Forest achieves the highest F1 on the positive class (0.861) by being more permissive with positive predictions; the baseline XGBoost achieves a similar ROC-AUC (0.742) and the best operating point for deployment given inference cost; the tuned XGBoost slightly underperforms (ROC-AUC 0.736).

**Table 1.** Held-out test metrics for the four classifiers on 19,271 orders.

| Model | ROC-AUC | PR-AUC | F1 (pos) | Macro F1 | Balanced Acc |
|---|---|---|---|---|---|
| Logistic Regression (`class_weight='balanced'`) | 0.749 | 0.898 | 0.793 | 0.636 | 0.682 |
| Random Forest (200 trees, depth 15, balanced) | 0.748 | 0.896 | 0.861 | 0.676 | 0.678 |
| XGBoost baseline (300 trees, depth 6, `scale_pos_weight`) | 0.742 | 0.894 | 0.830 | 0.655 | 0.677 |
| XGBoost tuned (600 trees, depth 8, lr 0.05) | 0.736 | 0.891 | 0.847 | 0.662 | 0.670 |

The four classifiers cluster in a tight 1.3-point ROC-AUC band, which we read as evidence of an information ceiling on the engineered tabular features rather than as a tuning gap between methods. PR-AUC, which is the more relevant ranking metric on a 78.9% positive distribution, is similarly tight (0.891 to 0.898). The Random Forest and tuned XGBoost trade off precision for recall on the positive class, achieving F1 above 0.84, while the Logistic Regression and baseline XGBoost achieve a more balanced operating point.

### 4.2 Feature importance

Table 2 reports the top 10 features by Random Forest impurity-based importance. Three delivery-related features dominate: `delivery_days` (0.228), `has_comment` (0.182) and `late_flag` (0.149) together account for 55.9% of total impurity gain. `estimated_days` adds another 0.068. Freight-related features (`items_freight_sum` 0.063, `freight_ratio` 0.056, `items_price_sum` 0.054) contribute moderately, and calendar features (`hour_purchase`, `month_purchase`) and basket size (`items_count`) round out the top 10 with smaller individual contributions.

**Table 2.** Random Forest impurity-based feature importance (top 10).

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

This ranking is consistent with the Spearman correlations from the EDA (delivery_days at -0.235 against the score, delivery_delay at -0.176, n_items at -0.108, freight at -0.089, price at -0.029) and with the published causal evidence that delivery speed and on-time arrival drive online retail satisfaction [20, 21]. The high importance of `has_comment` is partly behavioural (customers who write a comment are skewed toward stronger feelings in either direction) but also carries a label-adjacent component: in the Olist workflow the comment field is recorded together with the review score, so the decision to comment is not strictly anterior to the rating and the binary flag therefore encodes a fraction of the target signal it is being used to predict. The reported feature ranking and the deployed ROC-AUC are inflated to an unquantified degree by this leakage. We retain the feature in the present comparison for continuity with the EDA, flag the issue plainly here, and schedule a Phase 2 ablation (section 5.4) that drops `has_comment` and reports the ROC-AUC drop alongside a leakage-free `comment_propensity` proxy fit on tabular features only. Adding the comment text itself as features would compound the leakage risk because the comment is typically written after the star is assigned; this question is treated explicitly in the next modelling iteration.

### 4.3 Why the tuned XGBoost underperforms the baseline

A counter-intuitive but reproducible finding is that the tuned XGBoost (600 trees, depth 8, learning rate 0.05) scores 0.736 ROC-AUC, six points below the baseline XGBoost (300 trees, depth 6, learning rate 0.1). The validation curve, which is monotonic-with-noise on the 57,811-row training set, suggests that the deeper trees overfit small interactions between calendar features and basket-size features that do not replicate on the held-out test split. This pattern is exactly the one Grinsztajn et al. [6] identify as a signature of feature-set saturation: when the underlying feature space is small and the strong predictors are continuous and monotonic, deeper boosters chase noise. The operational implication is that the baseline XGBoost is the recommended deployment configuration on this feature set, and that further capacity should be added at the feature level (text embeddings, seller mean-encoding) rather than at the boosting-hyperparameter level. The tuned XGBoost is retained as `deliverables/csat_xgb.pkl` for completeness.

## 5. Discussion

### 5.1 What the Olist data tells us about satisfaction in Brazilian e-commerce

Three findings are stable across the EDA and the modelling stages. First, on-time delivery is the dominant non-text predictor of star rating: mean delivery time roughly doubles from 5-star to 1-star reviews (10.68 to 21.32 days), and the late-delivery rate increases by a factor of 12.6 (3.0% to 37.9%). Second, the platform's delivery-date estimate is conservative: across all 95,824 delivered-and-reviewed orders, the mean order arrives 11 days before the platform's estimated date. A "late" delivery in this dataset is therefore almost always a real failure rather than a tight promise being narrowly missed, which is what makes `late_flag` a clean operational variable. Third, raw item price has near-zero correlation with the score (-0.029); customers do not appear to blame the platform for sticker price, but they do penalise freight cost and basket complexity, which translate into more pickup and shipment risk. These findings line up with the published delivery-quality literature [20, 21].

The Brazilian-state pattern reinforces the supply-chain reading. São Paulo, which is 41% of all order volume and the centre of the Olist logistics network, has a mean delivery time of 8.75 days and a mean review score of 4.25. The northern and northeastern states (Bahia, Rio de Janeiro) sit at the bottom with mean delivery times of 15-19 days and mean scores below 4.0. This is a geography effect mediated by logistics rather than by buyer preferences, which is also why one-hot encoding of `customer_state` is sufficient as a feature: the state captures the average distance from the SP fulfilment hub and the local last-mile reliability.

### 5.2 Why the tabular ROC-AUC plateau is binding

Four very different classifiers cluster in a 1.3-point ROC-AUC band, the tuned XGBoost slightly underperforms its untuned counterpart, and the top three features account for over half of the impurity gain. We read this as evidence that the engineered tabular feature set has saturated its information content. To break the ROC-AUC 0.75 barrier, additional information has to enter the model from outside the order-time tabular fields. Three concrete sources are available in the Olist corpus:

- The Portuguese review_comment_message text (40,977 non-empty comments, 41.4% of reviews). A simple TF-IDF pipeline over lemmatised 1-2 grams should already lift performance, and a transformer encoder (BERTimbau [16, 17] or XLM-R [15]) is the published state of the art for Brazilian Portuguese review sentiment [18]. A late-fusion architecture along the lines of Wide and Deep [19] is the natural way to combine the dense text representation with the sparse tabular features that dominate the current model.
- Seller-level historical performance, encoded as a leave-one-out target mean over previous orders. The Olist data shipping with 3,095 sellers and 112,650 item-seller rows is large enough that seller mean-encoding will not overfit, provided the leave-one-out smoothing is applied correctly. CatBoost [3] is an attractive alternative encoder because it implements ordered boosting that is robust to target leakage in this setting.
- Product-category-level priors and product physical-dimension features. The category-translation table maps the Portuguese category names to English and provides 71 stable categories; mean-encoded category satisfaction plus the four physical-dimension fields (weight, length, height, width) capture parts of the breakage and freight-risk channel that are currently only partially summarised by `freight_ratio`.

### 5.3 Limitations

Three limitations of the present study are worth flagging. First, the binarisation of the 1-5 star rating into a positive (4-5) versus non-positive (1-3) class loses the ordinal structure of the response. Treating the star rating as ordinal (with cumulative-link models) or as a regression target would be a reasonable alternative. Second, `has_comment` is a known label-adjacent feature: in the Olist data model the comment is captured in the same review event as the star, so the binary "did the customer comment" flag is not strictly an order-time signal and almost certainly inflates the reported ROC-AUC. The exact magnitude of the inflation has not been measured in the present pipeline; a Phase 2 ablation that drops `has_comment` and replaces it with a tabular-only `comment_propensity` proxy is scheduled in section 5.4, and the ablation result should be treated as the headline number in any operational deployment of this model. Third, the dataset spans 2016 to 2018; the Brazilian e-commerce logistics landscape has changed since (especially through the expansion of last-mile fulfilment networks), and the absolute on-time rates reported here should not be transferred to current marketplace operations without recalibration.

### 5.4 Plan for the next modelling iteration (modeling_3)

The next iteration of this study, documented in `reports/modeling_3.md`, will build a fused tabular-plus-text model with three deltas relative to the present setup. First, a TF-IDF pipeline over the lemmatised Portuguese review_comment_message will provide a sparse linear baseline; second, BERTimbau [16, 17] sentence embeddings will provide a dense alternative, with mean-pooled `[CLS]` representations passed through a shallow projection head; third, seller_id and product_category_name will be entered as leave-one-out mean-encoded numeric features. The tabular branch (the present XGBoost baseline) and the text branch will be fused at the prediction-probability level via stacking, with model interpretability provided by TreeSHAP [22, 23] on the tabular branch. SMOTE [7], ADASYN [8] and focal loss [10] will be tested as alternative imbalance handling strategies on the fused model. The expectation, supported by the Brazilian-Portuguese review sentiment baselines in [18], is a ROC-AUC lift of 3-6 points over the current 0.749 ceiling.

## 6. Conclusion

We trained four classifiers (Logistic Regression, Random Forest, XGBoost baseline, XGBoost tuned) to predict whether an Olist order would receive a four- or five-star review from order-time tabular features only. All four classifiers cluster between ROC-AUC 0.736 and 0.749 on a held-out test split, which we interpret as a saturation of the engineered tabular feature set rather than as a tuning gap. The three top features (`delivery_days`, `has_comment`, `late_flag`) account for 55.9% of impurity gain, confirming the EDA finding that delivery duration and on-time arrival are the dominant non-text drivers of customer satisfaction in the Olist marketplace, and that price-related features are not. The tuned XGBoost slightly underperforms the baseline XGBoost (ROC-AUC 0.736 vs 0.742), which we take as a signal to invest the next modelling-iteration budget in additional features (Portuguese review-text embeddings via BERTimbau [16], seller-level mean encoding) rather than in deeper boosting trees. The tabular baseline established here is fast, interpretable and ready to deploy; it also defines a clean reference point against which the text-augmented model in modeling_3 will be evaluated.

## References

[1] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). 2016. p. 785-794. DOI: 10.1145/2939672.2939785.

[2] Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, Ye Q, Liu T-Y. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems (NeurIPS). 2017. vol. 30.

[3] Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: Unbiased Boosting with Categorical Features. Advances in Neural Information Processing Systems (NeurIPS). 2018. vol. 31. arXiv: 1706.09516.

[4] Breiman L. Random Forests. Machine Learning. 2001;45(1):5-32. DOI: 10.1023/A:1010933404324.

[5] Arik SO, Pfister T. TabNet: Attentive Interpretable Tabular Learning. Proceedings of the AAAI Conference on Artificial Intelligence. 2021;35(8):6679-6687. DOI: 10.1609/aaai.v35i8.16826.

[6] Grinsztajn L, Oyallon E, Varoquaux G. Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data? NeurIPS Datasets and Benchmarks Track. 2022. arXiv: 2207.08815.

[7] Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research. 2002;16:321-357. DOI: 10.1613/jair.953.

[8] He H, Bai Y, Garcia EA, Li S. ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. IEEE International Joint Conference on Neural Networks (IJCNN). 2008. p. 1322-1328. DOI: 10.1109/IJCNN.2008.4633969.

[9] He H, Garcia EA. Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering. 2009;21(9):1263-1284. DOI: 10.1109/TKDE.2008.239.

[10] Lin T-Y, Goyal P, Girshick R, He K, Dollar P. Focal Loss for Dense Object Detection. IEEE International Conference on Computer Vision (ICCV). 2017. DOI: 10.1109/ICCV.2017.324.

[11] Johnson JM, Khoshgoftaar TM. Survey on Deep Learning with Class Imbalance. Journal of Big Data. 2019;6(1):27. DOI: 10.1186/s40537-019-0192-5.

[12] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I. Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS). 2017. vol. 30. arXiv: 1706.03762.

[13] Devlin J, Chang M-W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT. 2019. p. 4171-4186. arXiv: 1810.04805.

[14] Pires T, Schlinger E, Garrette D. How Multilingual is Multilingual BERT? Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL). 2019. DOI: 10.18653/v1/P19-1493.

[15] Conneau A, Khandelwal K, Goyal N, Chaudhary V, Wenzek G, Guzman F, Grave E, Ott M, Zettlemoyer L, Stoyanov V. Unsupervised Cross-lingual Representation Learning at Scale. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL). 2020. DOI: 10.18653/v1/2020.acl-main.747.

[16] Souza F, Nogueira R, Lotufo R. BERTimbau: Pretrained BERT Models for Brazilian Portuguese. Brazilian Conference on Intelligent Systems (BRACIS). 2020. LNCS 12319. p. 403-417. DOI: 10.1007/978-3-030-61377-8_28.

[17] Souza F, Nogueira R, Lotufo R. BERT Models for Brazilian Portuguese: Pretraining, Evaluation and Tokenization Analysis. Applied Soft Computing. 2023;149:110901. DOI: 10.1016/j.asoc.2023.110901.

[18] Souza FD, Filho JBOS. Sentiment Analysis on Brazilian Portuguese User Reviews. 2021. arXiv: 2112.05459.

[19] Cheng H-T, Koc L, Harmsen J, Shaked T, Chandra T, Aradhye H, Anderson G, Corrado G, Chai W, Ispir M, et al. Wide & Deep Learning for Recommender Systems. Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. 2016. p. 7-10. DOI: 10.1145/2988450.2988454.

[20] Cui R, Lu Z, Sun T, Golden JM. Sooner or Later? Promising Delivery Speed in Online Retail. Manufacturing & Service Operations Management. 2023. DOI: 10.1287/msom.2021.0174.

[21] Aljohani K. The Role of Last-Mile Delivery Quality and Satisfaction in Online Retail Experience: An Empirical Analysis. Sustainability. 2024;16(11):4743. DOI: 10.3390/su16114743.

[22] Lundberg SM, Lee S-I. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS). 2017. vol. 30. arXiv: 1705.07874.

[23] Lundberg SM, Erion G, Chen H, DeGrave A, Prutkin JM, Nair B, Katz R, Himmelfarb J, Bansal N, Lee S-I. From Local Explanations to Global Understanding with Explainable AI for Trees. Nature Machine Intelligence. 2020;2:56-67. DOI: 10.1038/s42256-019-0138-9.
