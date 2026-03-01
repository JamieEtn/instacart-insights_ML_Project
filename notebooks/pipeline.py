# =============================================================================
# INSTACART ML PIPELINE 
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
# CONFIGURATION
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
# FOLDER SETUP
# =============================================================================
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print(" INSTACART RETAIL INSIGHTS PIPELINE")
print("=" * 60)
t0 = time.time()

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1/7] Loading data...")

orders = pd.read_csv(
    "data/raw/orders.csv",
    usecols=[
        "order_id", "user_id", "eval_set", 
        "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"
    ]
)

op_prior = pd.read_csv("data/raw/order_products__prior.csv", usecols=["order_id", "product_id", "reordered"])
op_train = pd.read_csv("data/raw/order_products__train.csv", usecols=["order_id", "product_id", "reordered"])
products = pd.read_csv("data/raw/products.csv", usecols=["product_id", "product_name", "aisle_id", "department_id"])
aisles = pd.read_csv("data/raw/aisles.csv")
departments = pd.read_csv("data/raw/departments.csv")

print("✓ Raw data loaded")

# =============================================================================
# 2. CLEAN & MERGE
# =============================================================================
print("\n[2/7] Cleaning & merging...")

# Merge prior and train products
order_products = pd.concat([op_prior, op_train], ignore_index=True)
del op_prior, op_train

# Keep only active users with enough orders
active_users = orders.groupby("user_id")["order_id"].count()
active_users = active_users[active_users >= MIN_ORDERS_PER_USER].index
orders = orders[orders["user_id"].isin(active_users)]

# Focus on train set
orders = orders[orders["eval_set"] == "train"]
order_products = order_products[order_products["order_id"].isin(orders["order_id"])]

# Merge product info
products = products.merge(aisles, on="aisle_id", how="left") \
                   .merge(departments, on="department_id", how="left")
products["aisle"] = products["aisle"].fillna("Unknown")
products["department"] = products["department"].fillna("Unknown")

# Build master transactional dataset
df = order_products.merge(
    orders[["order_id", "user_id", "order_dow", "order_hour_of_day", "days_since_prior_order"]],
    on="order_id", how="inner"
).merge(
    products[["product_id", "product_name", "aisle", "department"]],
    on="product_id", how="left"
)

df.to_csv("data/processed/master.csv", index=False)
print(f"✓ Master dataset saved ({len(df):,} rows)")

# =============================================================================
# 3. ASSOCIATION RULES (FP-GROWTH)
# =============================================================================
print("\n[3/7] Association rules (FP-Growth)...")

# Focus on top users and products for speed
top_users = df["user_id"].value_counts().head(TOP_USERS).index
top_products = df["product_name"].value_counts().head(TOP_PRODUCTS).index
basket_df = df[df["user_id"].isin(top_users) & df["product_name"].isin(top_products)]

# Aggregate transactions per order
transactions = basket_df.groupby("order_id")["product_name"].apply(list)

# One-hot encode
te = TransactionEncoder()
basket_encoded = pd.DataFrame.sparse.from_spmatrix(
    te.fit(transactions).transform(transactions, sparse=True),
    columns=te.columns_
)

# Frequent itemsets using FP-Growth
freq_items = fpgrowth(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)

if len(freq_items) > 0:
    rules = association_rules(freq_items, metric="lift", min_threshold=MIN_LIFT)
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE]
    rules = rules.nlargest(200, "lift")
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules.to_csv("results/association_rules.csv", index=False)
    print(f"✓ Association rules saved ({len(rules)})")
else:
    rules = pd.DataFrame()
    print("⚠ No frequent itemsets found")

# =============================================================================
# 3b. ALGORITHM TIMING COMPARISON
# =============================================================================
print("\n[3b/7] FP-Growth vs Apriori timing...")

algo_results = []

# FP-Growth
start = time.time()
freq_fp = fpgrowth(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)
algo_results.append({"Algorithm": "FP-Growth", "Itemsets": len(freq_fp), "Time (s)": round(time.time() - start, 2)})

# Apriori
start = time.time()
freq_ap = apriori(basket_encoded, min_support=MIN_SUPPORT, use_colnames=True)
algo_results.append({"Algorithm": "Apriori", "Itemsets": len(freq_ap), "Time (s)": round(time.time() - start, 2)})

pd.DataFrame(algo_results).to_csv("results/algorithm_comparison.csv", index=False)
print("✓ Algorithm comparison saved")

# =============================================================================
# 4. UTILITY-ORIENTED PATTERNS
# =============================================================================
print("\n[4/7] Utility mining approximation...")

# Assign price per department (seeded for reproducibility)
np.random.seed(101)
dept_prices = {d: np.random.uniform(1.5, 8.0) for d in products["department"].unique()}
products["est_price"] = products["department"].map(dept_prices).fillna(2.0)

df = df.merge(products[["product_id", "est_price"]], on="product_id", how="left")
df["est_price"] = df["est_price"].fillna(2.0)

# High-value orders
order_value = df.groupby("order_id")["est_price"].sum()
hv_threshold = order_value.quantile(UTILITY_QUANTILE)
hv_orders = order_value[order_value > hv_threshold].index
hv_basket = basket_df[basket_df["order_id"].isin(hv_orders)]

if hv_basket["order_id"].nunique() > 100:
    hv_trans = hv_basket.groupby("order_id")["product_name"].apply(list)
    te2 = TransactionEncoder()
    hv_encoded = pd.DataFrame.sparse.from_spmatrix(
        te2.fit(hv_trans).transform(hv_trans, sparse=True),
        columns=te2.columns_
    )
    freq_hv = fpgrowth(hv_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    if len(freq_hv) > 0:
        rules_hv = association_rules(freq_hv, metric="lift", min_threshold=MIN_LIFT)
        rules_hv["utility_score"] = rules_hv["support"] * rules_hv["confidence"] * rules_hv["lift"]
        rules_hv["itemset_str"] = rules_hv["antecedents"].apply(lambda x: ", ".join(sorted(x))) + " → " + \
                                  rules_hv["consequents"].apply(lambda x: ", ".join(sorted(x)))
        rules_hv = rules_hv.sort_values("utility_score", ascending=False).head(200)
        rules_hv.to_csv("results/utility_rules.csv", index=False)
        print(f"✓ Utility rules saved ({len(rules_hv)})")
    else:
        print("⚠ No high-utility itemsets found")
else:
    print("⚠ Not enough high-value orders for stable utility mining")

# =============================================================================
# 5. CUSTOMER SEGMENTATION (RFM + KMeans)
# =============================================================================
print("\n[5/7] Customer segmentation...")

# Frequency
freq = orders.groupby("user_id")["order_id"].count().rename("frequency")
# Recency = days since last order
recency = orders.groupby("user_id")["days_since_prior_order"].max().rename("recency")
# Monetary = total spend
monetary = df.groupby("user_id")["est_price"].sum().rename("monetary")

rfm = pd.concat([freq, recency, monetary], axis=1).reset_index()

# Standardize features
X = StandardScaler().fit_transform(rfm[["frequency", "recency", "monetary"]])

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
rfm["segment"] = kmeans.fit_predict(X)
rfm["buyer_type"] = np.where(rfm["frequency"] >= rfm["frequency"].median(), "Frequent", "Irregular")

rfm.to_csv("results/customer_segments.csv", index=False)
print("✓ Segmentation saved")

# =============================================================================
# 6. TOP PRODUCTS
# =============================================================================
print("\n[6/7] Top products ranking...")

top_prod = df["product_name"].value_counts().head(20).reset_index()
top_prod.columns = ["product_name", "order_count"]
top_prod.to_csv("results/top_products.csv", index=False)
print(f"✓ Top products saved ({len(top_prod)} products)")

# =============================================================================
# 6. REVENUE SIMULATION FROM BUNDLES
# =============================================================================
print("\n[7/8] Revenue simulation from bundles...")

if 'rules_hv' in globals() and len(rules_hv) > 0:
    bundle_revenue_data = []
    for _, row in rules_hv.iterrows():
        items = list(row['antecedents']) + list(row['consequents'])
        # Total estimated price of bundle
        bundle_price = df[df['product_name'].isin(items)].groupby('product_name')['est_price'].mean().sum()
        # Count how many orders include all items in the bundle
        orders_with_bundle = df.groupby('order_id')['product_name'].apply(set)
        n_orders = orders_with_bundle.apply(lambda x: set(items).issubset(x)).sum()
        total_revenue = bundle_price * n_orders

        bundle_revenue_data.append({
            'bundle': ", ".join(items),
            'bundle_price': round(bundle_price, 2),
            'n_orders': n_orders,
            'estimated_revenue': round(total_revenue, 0)
        })

    revenue_df = pd.DataFrame(bundle_revenue_data)
    revenue_df = revenue_df.sort_values('estimated_revenue', ascending=False).head(50)
    revenue_df.to_csv("results/bundle_revenue_simulation.csv", index=False)
    print(f"✓ Revenue simulation saved ({len(revenue_df)} bundles)")
else:
    print("⚠ No high-utility rules available for revenue simulation")

# =============================================================================
# 7. PROMOTION EFFICIENCY
# =============================================================================
print("\n[8/8] Promotion efficiency...")

promo_data = []
for seg in rfm["segment"].unique():
    seg_users = rfm[rfm["segment"] == seg]
    n_cust = len(seg_users)
    avg_spend = seg_users["monetary"].mean()
    
    blanket_cost = n_cust * avg_spend * 0.15
    targeted_cost = blanket_cost * 0.67
    targeted_gain = blanket_cost * 1.25
    advantage = (targeted_gain - targeted_cost) - (-blanket_cost)  # improvement over blanket

    promo_data.append({
        "segment": f"Segment {seg}",
        "customers": n_cust,
        "avg_spend": round(avg_spend, 1),
        "blanket_cost": round(blanket_cost, 0),
        "targeted_cost": round(targeted_cost, 0),
        "targeted_gain": round(targeted_gain, 0),
        "advantage": round(advantage, 0)
    })

promo_df = pd.DataFrame(promo_data)
promo_df.to_csv("results/promotion_efficiency.csv", index=False)
print("✓ Promotion efficiency saved")

# =============================================================================
# PIPELINE COMPLETE
# =============================================================================
print("\nPIPELINE COMPLETE")
print(f"⏱ Total time: {time.time() - t0:.1f}s")