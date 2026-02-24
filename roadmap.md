# Retail Insights Project — Roadmap
**Team of 7 | Instacart Dataset | Streamlit Dashboard**

---

## Team Role Split (recommended)

| Role | Person(s) | Focus |
|------|-----------|-------
| Data Engineer | 2 people | Data pipeline, merging, cleaning |
| Data Scientist A | 1 person | Association rules (Apriori, FP-Growth, Eclat) |
| Data Scientist B | 1 person | Customer segmentation + UP-Tree utility mining |
| Business Analyst | 1-2 people | Streamlit dashboard + report + demo video |

---

## Day 1 — Setup & Data Pipeline
**Owner: Data Engineer + everyone sets up env**

- [ ] Download Instacart dataset from Kaggle (see instructions below)
- [ ] Set up GitHub repo with branch-per-role structure
- [ ] Run `pipeline.py` to merge and clean all CSVs
- [ ] Explore dataset shape, nulls, dtypes
- [ ] Deliverable: `data/clean_orders.csv` ready for analysis

**Kaggle Download Steps:**
1. Go to https://www.kaggle.com/c/instacart-market-basket-analysis/data
2. Sign in and accept competition rules
3. Download all files: `orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`
4. Place in `data/raw/` folder

---

## Day 2 — EDA + Association Rule Mining
**Owner: Data Scientist A**

- [ ] Run EDA: order frequency, top products, reorder rates
- [ ] Build transaction matrix (basket format)
- [ ] Run Apriori algorithm → frequent itemsets
- [ ] Run FP-Growth algorithm → compare performance
- [ ] Run Eclat algorithm → compare
- [ ] Extract association rules (support, confidence, lift)
- [ ] Deliverable: `results/association_rules.csv`

---

## Day 3 — Customer Segmentation + Utility Mining
**Owner: Data Scientist B**

- [ ] RFM analysis (Recency, Frequency, Monetary)
- [ ] KMeans clustering → customer segments
- [ ] Label segments: Budget Shopper, Regular, Premium
- [ ] UP-Tree utility mining (value-aware bundles)
- [ ] Revenue simulation per bundle
- [ ] Deliverable: `results/customer_segments.csv`, `results/utility_rules.csv`

---

## Day 4 — Streamlit Dashboard
**Owner: Business Analyst(s)**

- [ ] Build Streamlit app with 4 pages:
  - Customer Segmentation
  - Product Associations & Bundles
  - Revenue Simulation
  - Promotion ROI Calculator
- [ ] Connect to results CSVs
- [ ] Add charts (plotly)
- [ ] Deliverable: working `app.py`

---

## Day 5 — Report + GitHub + Demo
**Owner: Everyone contributes**

- [ ] Write PDF report (template provided)
- [ ] Clean up GitHub repo (README, requirements.txt)
- [ ] Record short demo video (Loom or OBS — 3-5 min)
- [ ] Final review of notebook
- [ ] Submit!

---

## Project File Structure

```
retail-insights/
├── data/
│   ├── raw/          ← Kaggle CSV files go here
│   └── processed/    ← Auto-generated cleaned files
├── notebooks/
│   └── pipeline.ipynb   ← Main ML pipeline
├── results/
│   ├── association_rules.csv
│   ├── customer_segments.csv
│   └── utility_rules.csv
├── app.py               ← Streamlit dashboard
├── requirements.txt
└── README.md
```
