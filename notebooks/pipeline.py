# =============================================================================
# INSTACART ML PIPELINE — FINAL OPTIMIZED VERSION
# =============================================================================
import os
import time
import warnings

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# CONFIG
# =============================================================================
TOP_USERS = 5000
TOP_PRODUCTS = 500
MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.2
MIN_ORDERS_PER_USER = 3
UTILITY_QUANTILE = 0.75
N_CLUSTERS = 4

# =============================================================================
# FOLDERS
# =============================================================================
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print(" INSTACART RETAIL INSIGHTS PIPELINE")
print("=" * 60)
t0 = time.time()

# =============================================================================
# 1. LOAD DATA (MEMORY-AWARE)
# =============================================================================
print("\n[1/6] Loading data...")

orders = pd.read_csv(
    "data/raw/orders.csv",
    usecols=[
        "order_id",
        "user_id",
        "eval_set",
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    ],
)

op_prior = pd.read_csv(
    "data/raw/order_products__prior.csv",
    usecols=["order_id", "product_id", "reordered"],
)

op_train = pd.read_csv(
    "data/raw/order_products__train.csv",
    usecols=["order_id", "product_id", "reordered"],
)

products = pd.read_csv(
    "data/raw/products.csv",
    usecols=["product_id", "product_name", "aisle_id", "department_id"],
)

aisles = pd.read_csv("data/raw/aisles.csv")
departments = pd.read_csv("data/raw/departments.csv")

print("✓ Raw data loaded")

# =============================================================================
# 2. CLEAN + MERGE
# =============================================================================
print("\n[2/6] Cleaning & merging...")

# Combine prior + train order-product links
order_products = pd.concat([op_prior, op_train], ignore_index=True)
del op_prior, op_train

# Keep only "active" users with enough history
active_users = (
    orders.groupby("user_id")["order_id"]
    .count()
    .loc[lambda x: x >= MIN_ORDERS_PER_USER]
    .index
)
orders = orders[orders["user_id"].isin(active_users)]

# Focus on "train" orders (so we work on a consistent subset)
orders = orders[orders["eval_set"] == "train"]

# Keep only lines belonging to these orders
order_products = order_products[order_products["order_id"].isin(orders["order_id"])]

# Enrich products with aisle/department names
products = (
    products.merge(aisles, on="aisle_id", how="left")
    .merge(departments, on="department_id", how="left")
)

products["aisle"] = products["aisle"].fillna("Unknown")
products["department"] = products["department"].fillna("Unknown")

# Master transactional dataset (one line = one product in one order)
df = (
    order_products.merge(
        orders[
            [
                "order_id",
                "user_id",
                "order_dow",
                "order_hour_of_day",
                "days_since_prior_order",
            ]
        ],
        on="order_id",
        how="inner",
    )
    .merge(
        products[["product_id", "product_name", "aisle", "department"]],
        on="product_id",
        how="inner",
    )
)

df.to_csv("data/processed/master.csv", index=False)
print(f"✓ Master dataset saved ({len(df):,} rows)")

# =============================================================================
# 3. ASSOCIATION RULES (FP-GROWTH)
# =============================================================================
print("\n[3/6] Association mining (FP-Growth)...")

# Focus on top users / products to keep it fast and interpretable
top_users = df["user_id"].value_counts().head(TOP_USERS).index
top_products = df["product_name"].value_counts().head(TOP_PRODUCTS).index

basket_df = df[
    (df["user_id"].isin(top_users)) & (df["product_name"].isin(top_products))
]

# One "transaction" = one order -> list of product names
transactions = basket_df.groupby("order_id")["product_name"].apply(list)

# One-hot encode transactions for fpgrowth
te = TransactionEncoder()
basket_encoded = pd.DataFrame.sparse.from_spmatrix(
    te.fit(transactions).transform(transactions, sparse=True),
    columns=te.columns_,
)

# Frequent itemsets with FP-Growth
freq_items = fpgrowth(
    basket_encoded,
    min_support=MIN_SUPPORT,
    use_colnames=True,
)

if len(freq_items) > 0:
    rules = association_rules(
        freq_items,
        metric="lift",
        min_threshold=MIN_LIFT,
    )

    # Apply confidence filter and keep top rules by lift
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE]
    rules = rules.nlargest(200, "lift")

    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(x))
    )

    rules.to_csv("results/association_rules.csv", index=False)
    print(f"✓ Association rules saved ({len(rules)})")
else:
    rules = pd.DataFrame()
    print("⚠ No frequent itemsets found")

# -----------------------------------------------------------------------------
# 3b. ALGORITHM COMPARISON: FP-Growth vs Apriori
# -----------------------------------------------------------------------------
print("\n[3b/6] Algorithm comparison (FP-Growth vs Apriori)...")

algo_results = []

# FP-Growth timing
start = time.time()
freq_fp = fpgrowth(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)
fp_time = time.time() - start
algo_results.append(
    {
        "Algorithm": "FP-Growth",
        "Itemsets": len(freq_fp),
        "Time (s)": round(fp_time, 2),
    }
)

# Apriori timing
start = time.time()
freq_ap = apriori(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)
ap_time = time.time() - start
algo_results.append(
    {
        "Algorithm": "Apriori",
        "Itemsets": len(freq_ap),
        "Time (s)": round(ap_time, 2),
    }
)

pd.DataFrame(algo_results).to_csv("results/algorithm_comparison.csv", index=False)
print("✓ Algorithm comparison saved")

# =============================================================================
# 4. UTILITY-ORIENTED PATTERNS (APPROXIMATION)
# =============================================================================
print("\n[4/6] Utility mining (approximate)...")

# 1) Assign a realistic price per department
dept_prices = {
    d: np.random.uniform(1.5, 8.0) for d in products["department"].unique()
}
products["est_price"] = products["department"].map(dept_prices)

df = df.merge(products[["product_id", "est_price"]], on="product_id", how="left")
df["est_price"] = df["est_price"].fillna(2.0)

# 2) Identify "high-value" orders by total basket value
order_value = df.groupby("order_id")["est_price"].sum()
threshold = order_value.quantile(UTILITY_QUANTILE)
hv_orders = order_value[order_value > threshold].index

hv_basket = basket_df[basket_df["order_id"].isin(hv_orders)]

if hv_basket["order_id"].nunique() > 100:
    hv_trans = hv_basket.groupby("order_id")["product_name"].apply(list)

    te2 = TransactionEncoder()
    hv_encoded = pd.DataFrame.sparse.from_spmatrix(
        te2.fit(hv_trans).transform(hv_trans, sparse=True),
        columns=te2.columns_,
    )

    freq_hv = fpgrowth(hv_encoded, min_support=MIN_SUPPORT, use_colnames=True)

    if len(freq_hv) > 0:
        rules_hv = association_rules(
            freq_hv, metric="lift", min_threshold=MIN_LIFT
        )

        # Utility score = support × confidence × lift
        rules_hv["utility_score"] = (
            rules_hv["support"]
            * rules_hv["confidence"]
            * rules_hv["lift"]
        )

        rules_hv["itemset_str"] = (
            rules_hv["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            + " → "
            + rules_hv["consequents"].apply(lambda x: ", ".join(sorted(x)))
        )

        # Keep the top rules by utility score
        rules_hv = rules_hv.sort_values("utility_score", ascending=False).head(200)

        rules_hv.to_csv("results/utility_rules.csv", index=False)
        print(f"✓ Utility rules saved ({len(rules_hv)})")
    else:
        print("⚠ No high-utility itemsets found (frequent itemsets empty)")
else:
    print("⚠ Not enough high-value orders for stable utility mining")

# =============================================================================
# 5. CUSTOMER SEGMENTATION (RFM + K-MEANS)
# =============================================================================
print("\n[5/6] Customer segmentation (RFM + KMeans)...")

# RFM features
freq = orders.groupby("user_id")["order_id"].count().rename("frequency")
rec = orders.groupby("user_id")["days_since_prior_order"].mean().rename("recency")
mon = df.groupby("user_id")["est_price"].sum().rename("monetary")

rfm = pd.concat([freq, rec, mon], axis=1).reset_index()

# Scale features (best practice before KMeans)[web:15][web:17]
X = StandardScaler().fit_transform(rfm[["frequency", "recency", "monetary"]])

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
rfm["segment"] = kmeans.fit_predict(X)

# Simple business-friendly tag: frequent vs irregular buyers
rfm["buyer_type"] = np.where(
    rfm["frequency"] >= rfm["frequency"].median(), "Frequent", "Irregular"
)

rfm.to_csv("results/customer_segments.csv", index=False)
print("✓ Segmentation saved")

# =============================================================================
# 6. REVENUE SIMULATION FROM BUNDLES
# =============================================================================
print("\n[6/6] Revenue simulation...")

if not rules.empty:
    # We already have rules DataFrame in memory
    avg_order = df.groupby("order_id")["est_price"].sum().mean()
    n_customers = rfm["user_id"].nunique()

    # Estimate how many new orders each rule could influence
    rules["est_new_orders"] = (
        rules["support"] * n_customers * rules["confidence"]
    ).astype(int)

    # Revenue gain from lift, minus the cost of a 10% discount campaign
    rules["revenue_from_lift"] = (
        rules["est_new_orders"] * avg_order * (rules["lift"] - 1)
    )
    rules["discount_cost"] = rules["est_new_orders"] * avg_order * 0.10
    rules["net_revenue_gain"] = (
        rules["revenue_from_lift"] - rules["discount_cost"]
    )

    # Characterize bundle size and type
    rules["bundle_size"] = (
        rules["antecedents"].apply(len)
        + rules["consequents"].apply(len)
    )
    rules["bundle_type"] = np.where(
        rules["bundle_size"] == 2, "Simple Pair", "Multi-item Bundle"
    )

    rules.to_csv("results/revenue_simulation.csv", index=False)
    print("✓ Revenue simulation saved")
else:
    print("⚠ Revenue simulation skipped (no association rules)")

print("\n PIPELINE COMPLETE")
print(f"⏱ Total time: {time.time() - t0:.1f}s")
