# Additional References - Olist Supply-Chain Customer Satisfaction

Independent literature scout pass. Each entry was resolved live against
`https://api.crossref.org/works/{doi}` on 2026-05-08. Papers that did not
resolve, or whose CrossRef record did not match the title/authors, were
omitted (no padding).

Sub-topic groupings are non-exclusive. Format: `Authors. Title. Journal. Year. DOI:...`
(no volume/issue/pages, per project policy).

---

## State-of-the-art callout - gaps in the current `references.md`

The existing `manuscripts/references.md` (23 entries) covers GBDT, class-imbalance
classics, transformer NLP, BERTimbau, Wide and Deep, delivery causal evidence
(Cui 2023, Aljohani 2024), and SHAP. After a final cross-check against that
file, the most material gaps - and what the project should cite next - are:

1. **TabPFN family (in-context tabular foundation models, 2024-2026).** The
   manuscript cites Grinsztajn 2022 to justify "GBDTs over deep tabular".
   That conclusion is now contested by TabPFN-class models. Adding Helli et al.
   2024 (Drift-Resilient TabPFN, NeurIPS) and Somvanshi et al. 2026 (ACM
   Computing Surveys) gives the reader an up-to-date picture and lets the
   discussion explicitly defend the XGBoost choice on a 96k-row / 14-feature
   problem.
2. **Direct precedent on tabular text-plus-tabular fusion for review rating.**
   Bonnier 2024 (ACL Findings) revisits multimodal transformers for tabular
   data with text fields. This is the closest published precedent for the
   modeling_3 plan (TF-IDF or BERTimbau plus tabular fusion) and should be
   cited where `[19] Wide and Deep` is currently the only fusion citation.
2.  **Direct Olist-on-Olist machine learning baselines (2025-2026).** The
   manuscript treats Olist as a fresh dataset; Yanuar and Princess 2026
   (ISIBER) and Garg 2025 (Preprints) report customer-segmentation and
   review-rating prediction baselines on Olist or near-equivalent fashion
   e-commerce data. These should anchor the "comparison with prior work"
   sentence in the discussion.
3.  **2024-2025 BERTimbau evidence on Brazilian Portuguese sentiment.**
   Schuck et al. 2025 (J. Brazilian Computer Society) and Walczak et al.
   2025 (HLT) post-date the manuscript's BERTimbau citations (refs 16-18,
   2020-2023) and report stronger numbers on Portuguese review sentiment;
   they should be cited next to ref [18] in the modeling_3 plan.
4.  **Last-mile and delivery causal evidence after Aljohani 2024.** Qu and
   Wang 2026 (Cleaner Logistics and Supply Chain) and Nguyen and Nguyen 2025
   (J. Finance and Accounting Research) both add post-2024 evidence that
   last-mile quality drives satisfaction and loyalty; useful as companions
   to refs [20] and [21] in section 1 and 5.1.

---

## A. Olist or directly comparable e-commerce ML baselines (2024-2026)

Yanuar K, Princess E. E-Commerce Customer and Transaction Behavior Segmentation Analysis Using RFM, Clustering, and Classification: A Case Study of Tokopedia and Olist (Brazil). 2026 International Seminar on Intelligent Business and Edge-Computing Research (ISIBER). 2026. DOI:10.1109/isiber68248.2026.11470484

Garg J. Predicting Customer Review Ratings in Fashion E-Commerce: A Machine Learning Approach with Enhanced Feature Engineering. Preprints. 2025. DOI:10.20944/preprints202510.1244.v1

Jahan I, Sanam TF. A comprehensive framework for customer retention in E-commerce using machine learning based on churn prediction, customer segmentation, and recommendation. Electronic Commerce Research. 2026. DOI:10.1007/s10660-024-09936-0

Wang W. An IoT-Based Framework for Cross-Border E-Commerce Supply Chain Using Machine Learning and Optimization. IEEE Access. 2024. DOI:10.1109/access.2023.3347452

## B. Tabular ML, gradient boosting, and tabular foundation models (2024-2026)

Somvanshi S, Das S, Javed S, Antariksa G, Hossain A. A Survey on Tabular Data: From Tree-based Methods to Tabular Deep Learning. ACM Computing Surveys. 2026. DOI:10.1145/3807777

Lazar A, Pokhrel P, Das S. Beyond Accuracy: A Comprehensive Comparative Study of Gradient Boosting Versus Tabular Deep Learning and Explainability Techniques for Mixed-Type Tabular Data Models Using SHAP and LIME. International Journal on Artificial Intelligence Tools. 2026. DOI:10.1142/s0218213026400038

Raparthi M, Dhabliya D, Kumari T, Upadhyaya R, Sharma A. Implementation and Performance Comparison of Gradient Boosting Algorithms for Tabular Data Classification. Algorithms for Intelligent Systems (ICDLAI). 2024. DOI:10.1007/978-981-97-4533-3_36

Helli K, Hollmann N, Hutter F, Mueller S, Schnurr D. Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data. Advances in Neural Information Processing Systems (NeurIPS). 2024. DOI:10.52202/079017-3134

Somvanshi S, Hebli P, Chhetri G, Das S. Tabular Data with Class Imbalance: Predicting Electric Vehicle Crash Severity with Pretrained Transformers (TabPFN) and Mamba-Based Models. 2025 International Conference on Machine Learning and Applications (ICMLA). 2025. DOI:10.1109/icmla66185.2025.00222

## C. Multimodal text-plus-tabular fusion (2024-2025)

Bonnier T. Revisiting Multimodal Transformers for Tabular Data with Text Fields. Findings of the Association for Computational Linguistics ACL 2024. 2024. DOI:10.18653/v1/2024.findings-acl.87

Aljuhani H, Dahab MY, Alsenani Y. Enhancing Document Classification Through Multimodal Image-Text Classification: Insights from Fine-Tuned CLIP and Multimodal Deep Fusion. Sensors. 2025. DOI:10.3390/s25247596

## D. Brazilian Portuguese sentiment and BERTimbau (2023-2025)

Schuck AdaF, Garcia GL, Manesco JRR, Paiola PH, Papa JP. Evaluating Large Language Models for Brazilian Portuguese Sentiment Analysis: A Comparative Study of Multilingual State-of-the-Art vs. Brazilian Portuguese Fine-Tuned LLMs. Journal of the Brazilian Computer Society. 2025. DOI:10.5753/jbcs.2025.5793

Walczak A, Kondratowicz P, Jaworski R. Assessing the Sentiment and Value of Product Reviews in Brazilian Portuguese. Human Language Technology as a Challenge for Computer Science and Linguistics. 2025. DOI:10.14746/amup.9788323245414.17

Da Rocha Junqueira J, Da Silva F, Costa W, Carvalho R, Bender A, Correa U, Freitas L. BERTimbau in Action: An Investigation of its Abilities in Sentiment Analysis, Aspect Extraction, Hate Speech Detection, and Irony Detection. The International FLAIRS Conference Proceedings. 2023. DOI:10.32473/flairs.36.133186

Oliveira FB, Sichman JS. Portuguese Emotion Detection Model Using BERTimbau Applied to COVID-19 News and Replies. Lecture Notes in Computer Science (Intelligent Systems). 2025. DOI:10.1007/978-3-031-79032-4_19

## E. Last-mile delivery and customer satisfaction (2024-2026)

Qu C, Wang K. How to improve the system evaluation of last-mile delivery? A strategy based on customer satisfaction model and spatial-time optimization. Cleaner Logistics and Supply Chain. 2026. DOI:10.1016/j.clscn.2026.100296

Nguyen HH, Nguyen HN. The mediating role of customer satisfaction on the impact of last mile delivery quality on e-commerce customer loyalty in Vietnam. Journal of Finance and Accounting Research. 2025. DOI:10.71374/jfar.v25.i4.18

Mogire E, Kilbourn P, Luke R. Last Mile Delivery Technologies for Electronic Commerce: A Bibliometric Review. Journal of Electronic Commerce in Organizations. 2025. DOI:10.4018/jeco.385122

Shamout MD, Alzoubi HM, Elrehail H, Itani R, Al-Gasaymeh A, Farouk M. Streamlining E-Commerce and Last-Mile Delivery for Enhanced Customer Satisfaction: An In-Depth Analysis of Amazon Strategies. 2024 2nd International Conference on Cyber Resilience (ICCR). 2024. DOI:10.1109/iccr61006.2024.10532856

Santosh L, Shetty R. A Study on the Impact of Last-Mile Delivery Efficiency on Customer Satisfaction and Cost Optimization in Quick Commerce. EPRA International Journal of Multidisciplinary Research. 2025. DOI:10.36713/epra22793

## F. Supply-chain machine learning and abnormal-delivery detection (2024-2026)

Eichenseer P, Hans L, Winkler H. A data-driven machine learning model for forecasting delivery positions in logistics for workforce planning. Supply Chain Analytics. 2025. DOI:10.1016/j.sca.2024.100099

Rajendran B, Gunasekaran A, Babu M. A machine learning framework for classifying customer advocacy in sustainable supply chains. Supply Chain Analytics. 2025. DOI:10.1016/j.sca.2025.100137

Ziabari G. Machine Learning to Detect Abnormal Delivery Performance in Supply Chain Operations. Preprints. 2025. DOI:10.20944/preprints202512.1746.v1

## G. Class imbalance applied to customer review classification (2024)

Li Z, Shimada K. Addressing Class Imbalance in Customer Review Analysis Using Focal Loss and SVM with BERT. 2024 17th International Congress on Advanced Applied Informatics (IIAI-AAI-Winter). 2024. DOI:10.1109/iiai-aai-winter65925.2024.00020

---

## Verification log

Each DOI above was confirmed live via `https://api.crossref.org/works/{doi}`
on 2026-05-08. Title and author match was checked against the CrossRef
record before the entry was kept. Items that returned HTTP 404, that
returned a non-matching title, or that were not 2023-2026 were dropped
during the screen and never appear in the list above.

24 verified entries across seven sub-topic groups.
