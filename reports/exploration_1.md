# Exploration Report 1: Olist Supply Chain Customer Satisfaction

**Project:** Liora #2 (Supply Chain - Customer Satisfaction)
**Date:** 2026-05-01
**Notebook:** `notebooks/01_eda.ipynb`

## 1. Project framing

The DataScientest brief asks for a customer-satisfaction study built on review data, with the explicit modelling target being a star-rating regression / classification (1-5) plus optional NLP on the verbatim comment. The brief points to scraping Trustpilot or using Trusted Shops verified reviews. The dataset supplied for this Liora project is the public **Olist Brazilian e-commerce** corpus (2016-2018), which is a stronger fit for a supply-chain framing than scraped Trustpilot stars: every review is tied to a verified purchase with the full upstream trail (seller, items, shipment timestamps, freight cost, payment, delivery date), so we can actually measure the supply chain's effect on satisfaction rather than just the post-hoc opinion.

## 2. Dataset description

Nine CSV files, ~164 MB on disk. They form a star schema centred on `orders`:

```
customers --< orders >-- order_reviews
                |   \--< order_items >-- products --- category_translation
                |       \-------------- sellers
                \--< order_payments
geolocation (zip_prefix, lat/lng)  attaches to customers and sellers via zip prefix
```

### Row counts and column counts

| Table | Rows | Cols | Role |
|---|---:|---:|---|
| `olist_orders_dataset` | 99,441 | 8 | Fact table (one row per order) |
| `olist_order_items_dataset` | 112,650 | 7 | One row per (order, item, seller) |
| `olist_order_payments_dataset` | 103,886 | 5 | One row per (order, payment leg) |
| `olist_order_reviews_dataset` | 99,224 | 7 | One review per order (a few orders have >1) |
| `olist_customers_dataset` | 99,441 | 5 | One row per `customer_id` (unique buyer-per-order) |
| `olist_products_dataset` | 32,951 | 9 | Product catalog with category and physical dims |
| `olist_sellers_dataset` | 3,095 | 4 | Seller directory with zip prefix and city/state |
| `olist_geolocation_dataset` | 1,000,163 | 5 | Brazilian zip-prefix lat/lng lookup |
| `product_category_name_translation` | 71 | 2 | PT to EN category mapping |

### Schemas (key columns)

- **orders:** `order_id`, `customer_id`, `order_status` (delivered / shipped / canceled / unavailable / invoiced / processing / created / approved), and 5 timestamp columns (`order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date`). All timestamps load as strings and need parsing.
- **order_items:** `order_id`, `order_item_id`, `product_id`, `seller_id`, `shipping_limit_date`, `price`, `freight_value`.
- **order_payments:** `order_id`, `payment_sequential`, `payment_type` (credit_card / boleto / voucher / debit_card / not_defined), `payment_installments`, `payment_value`.
- **order_reviews:** `review_id`, `order_id`, `review_score` (int 1-5, **the target**), `review_comment_title`, `review_comment_message`, `review_creation_date`, `review_answer_timestamp`.
- **customers:** `customer_id`, `customer_unique_id`, `customer_zip_code_prefix`, `customer_city`, `customer_state`. Note: `customer_id` is per-order; `customer_unique_id` is the real shopper.
- **products:** `product_id`, `product_category_name` (Portuguese), 4 description-length / photo-count fields, 4 physical-dimension fields.
- **sellers:** `seller_id`, `seller_zip_code_prefix`, `seller_city`, `seller_state`.

### Order status distribution

| status | count |
|---|---:|
| delivered | 96,478 |
| shipped | 1,107 |
| canceled | 625 |
| unavailable | 609 |
| invoiced | 314 |
| processing | 301 |
| created | 5 |
| approved | 2 |

97% of orders are delivered. We will restrict the modelling cohort to `order_status == 'delivered'` to keep delivery-time features well-defined.

## 3. Missing values per table

| table | column | n_missing | pct_missing |
|---|---|---:|---:|
| orders | order_approved_at | 160 | 0.16% |
| orders | order_delivered_carrier_date | 1,783 | 1.79% |
| orders | order_delivered_customer_date | 2,965 | 2.98% |
| order_reviews | review_comment_title | 87,656 | 88.34% |
| order_reviews | review_comment_message | 58,247 | 58.70% |
| products | product_category_name | 610 | 1.85% |
| products | product_name_lenght / description_lenght / photos_qty | 610 | 1.85% |
| products | product_weight_g / length_cm / height_cm / width_cm | 2 | 0.006% |
| order_items, order_payments, customers, sellers, geolocation, category_translation | (none) | 0 | 0.00% |

The missing delivery timestamps line up almost exactly with the non-delivered statuses (canceled, unavailable, shipped-but-not-yet-delivered), so this missingness is structural rather than noise. The big one is `review_comment_message`: 58.7% of reviews carry only a star, no text. That still leaves **40,977 reviews with a written comment**, which is enough for downstream NLP on the verbatim.

## 4. Target distribution: `review_score` (1-5)

| score | count | pct |
|---:|---:|---:|
| 1 | 11,424 | 11.51% |
| 2 | 3,151 | 3.18% |
| 3 | 8,179 | 8.24% |
| 4 | 19,142 | 19.29% |
| 5 | 57,328 | 57.78% |

Very heavy positive skew: 5-star alone is ~58% of all reviews, and 4 or 5 together is ~77%. Mean score is 4.09. This is a textbook imbalanced-classification setup, and most useful real-signal lives in the 1-3 tail (the dissatisfied customers, ~23% of reviews). For the modelling stage we will most likely binarise (positive: 4-5, negative: 1-3) in addition to the full ordinal target.

After joining reviews onto orders we get `review_score` coverage on **98,673 / 99,441 orders (99.2%)**, so review-attrition is negligible.

## 5. Delivery time vs satisfaction (key supply-chain finding)

After parsing timestamps we build:
- `delivery_days` = customer-delivery date - purchase timestamp
- `estimated_days` = estimated-delivery date - purchase timestamp
- `delivery_delay` = `delivery_days` - `estimated_days` (positive = late, negative = early)

On the 95,824 delivered orders that also have a review:

| review_score | n | mean delivery days | median delivery days | mean delay (days) | % late |
|---:|---:|---:|---:|---:|---:|
| 1 | 9,340 | 21.32 | 16.79 | -3.35 | **37.9%** |
| 2 | 2,922 | 16.68 | 13.27 | -7.93 | 20.6% |
| 3 | 7,903 | 14.25 | 12.04 | -10.09 | 11.0% |
| 4 | 18,894 | 12.31 | 10.79 | -11.68 | 5.0% |
| 5 | 56,765 | 10.68 | 9.23 | -12.68 | **3.0%** |

This is the cleanest single signal in the whole dataset. Delivery time rises monotonically as score drops, and the late-delivery rate jumps from **3% on 5-star orders to 38% on 1-star orders, more than a 12x ratio**. Olist clearly under-promises on the estimate (everyone is on average ~11 days early), which means a "late" delivery is almost always a real screw-up, not a tight deadline.

## 6. Numeric correlations with `review_score`

Spearman correlations (delivered orders, n=95,824):

| feature | spearman |
|---|---:|
| delivery_days | **-0.235** |
| delivery_delay | -0.176 |
| n_items | -0.108 |
| n_sellers | -0.095 |
| n_products | -0.090 |
| freight_sum | -0.089 |
| estimated_days | -0.056 |
| payment_total | -0.040 |
| price_sum | -0.029 |
| max_installments | -0.021 |

Delivery duration dominates. Basket complexity (number of items, number of distinct sellers, freight cost) gives a small but consistent negative pull on score. Raw price and installment count barely matter, which suggests price-based dissatisfaction is a non-issue on this platform: cheap and expensive baskets get rated about the same.

## 7. Geography and category snapshots

**Top-10 customer states by volume** (delivered, with review):

| state | n | mean score | mean delivery days |
|---|---:|---:|---:|
| SP | 40,266 | 4.25 | 8.75 |
| RJ | 12,211 | 3.96 | 15.24 |
| MG | 11,285 | 4.19 | 11.98 |
| RS | 5,326 | 4.19 | 15.28 |
| PR | 4,900 | 4.24 | 11.97 |
| SC | 3,519 | 4.13 | 14.88 |
| BA | 3,229 | 3.93 | 19.25 |
| DF | 2,070 | 4.14 | 12.96 |
| ES | 1,969 | 4.08 | 15.61 |
| GO | 1,946 | 4.10 | 15.56 |

São Paulo (SP) is 41% of all volume and gets the fastest deliveries (8.75 days mean) and the highest mean score (4.25). Bahia (BA) and Rio de Janeiro (RJ) sit at the bottom of the score distribution with the slowest deliveries. The state-level pattern is consistent with the delivery-time finding: distance from the SP logistics hub costs both speed and stars.

**Lowest-scoring categories (n >= 500 reviews):** office_furniture (3.49), bed_bath_table (3.90), furniture_decor (3.91), computers_accessories (3.93), home_construction (3.94).

**Highest-scoring categories (n >= 500 reviews):** perfumery (4.17), pet_shop (4.18), stationery (4.19), luggage_accessories (4.32), books_general_interest (4.45).

The bottom of the table is dominated by bulky furniture / home goods, which fits the delivery-time story (heavier, slower, more breakage). The top is small-parcel categories that ship fast and arrive intact.

## 8. Key observations

1. **Star schema is clean.** All foreign keys join cleanly, 99.2% review coverage on orders, 97% of orders reach `delivered` status. Almost nothing has to be thrown away upfront.
2. **Target is heavily imbalanced toward 5 stars** (58%), and the dissatisfied tail (1-3) is ~23%. Modelling needs class balancing (binary positive/negative target, weighted loss, or resampling).
3. **Delivery time is the dominant non-text predictor.** Mean delivery days for 1-star vs 5-star is 21.3 vs 10.7. Late-delivery rate is 12x higher on 1-star orders. Spearman with score is -0.24 for delivery days.
4. **Comment text is sparse but usable.** ~41,000 of 99,000 reviews carry a written message (Portuguese). This is the verbatim channel the DataScientest brief asks us to mine for delivery-problem / defect / price complaints.
5. **Price is not a satisfaction driver.** Raw item price has near-zero correlation with score. Freight value and basket size do show a small negative correlation, suggesting customers blame the platform for shipping pain, not for sticker price.
6. **Geography and category effects are real and align with logistics.** SP / fast delivery / small-parcel categories cluster at the high-score end; northern states / bulky furniture cluster at the low-score end.

## 9. Implications for the next steps

- **Modelling cohort:** restrict to `order_status == 'delivered'` with non-null delivery date and a review score (n ~ 96k).
- **Tabular features for the regression / classification arm:** delivery_days, delivery_delay, freight ratio (freight_sum / price_sum), n_items, n_sellers, product category (one-hot or target-encoded), customer state, payment type, installments.
- **NLP arm:** TF-IDF + linear baseline on `review_comment_message` (Portuguese), then sentence-transformer embeddings for the comparison. Translate to English only if needed for downstream entity extraction.
- **Targets to evaluate:** binary (1-3 vs 4-5) for a clean decision boundary, plus the full 1-5 ordinal as a secondary regression target.
- **Class imbalance:** use `class_weight='balanced'` or stratified resampling; report PR-AUC, not just accuracy.
