# =============================================================================
# RETAIL INSIGHTS DASHBOARD — Streamlit App
# Run with: streamlit run app.py
# =============================================================================

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Retail Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
        border-radius: 10px;
        padding: 16px 18px;
        color: white;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 1.8em; font-weight: 600; }
    .metric-label { font-size: 0.85em; opacity: 0.9; margin-top: 4px; }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #2b6cb0;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.95em;
    }
    h1, h2, h3 { color: #2d3748; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all precomputed CSVs from the ML pipeline."""
    data = {}
    files = {
        "rules": "results/association_rules.csv",
        "segments": "results/customer_segments.csv",
        "revenue": "results/revenue_simulation.csv",
        "promotion": "results/promotion_efficiency.csv",
        "top_prod": "results/top_products.csv",
        "algo_comp": "results/algorithm_comparison.csv",
        "reorder": "results/reorder_by_department.csv",
        "utility": "results/utility_rules.csv",
    }
    for key, path in files.items():
        data[key] = pd.read_csv(path) if os.path.exists(path) else None
    return data


data = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Retail Insights")
    st.markdown("Data‑Driven Cost Savings & Revenue Growth")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Customer Segments",
            "Product Associations",
            "Revenue Simulation",
            "Promotion ROI",
        ],
    )

    st.divider()
    st.caption("Instacart Online Grocery Dataset\n3M+ orders | 200K+ customers")

# ─── Helper: check data ───────────────────────────────────────────────────────
def data_missing(key, fname):
    if data[key] is None:
        st.warning(f"Run `pipeline.py` first to generate `{fname}`.")
        return True
    return False


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if page == "Overview":
    st.title("Retail Insights Dashboard")
    st.markdown("Data‑driven analysis of customer behaviour, product bundles and revenue impact.")
    st.divider()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    n_customers = (
        data["segments"]["user_id"].nunique() if data["segments"] is not None else None
    )
    n_rules = len(data["rules"]) if data["rules"] is not None else None

    if data["revenue"] is not None:
        total_gain = data["revenue"]["net_revenue_gain"].sum()
        gain_str = f"${total_gain:,.0f}"
    else:
        gain_str = "N/A"

    if data["promotion"] is not None:
        promo_overview = data["promotion"].copy()
        # CSV has column "advantage"
        if "advantage" in promo_overview.columns:
            targeting_adv = promo_overview["advantage"].sum()
            adv_str = f"${targeting_adv:,.0f}"
        else:
            adv_str = "N/A"
    else:
        adv_str = "N/A"

    with col1:
        st.metric(
            "Customers analysed",
            f"{n_customers:,}" if isinstance(n_customers, (int, np.integer)) else "N/A",
        )
    with col2:
        st.metric(
            "Association rules",
            f"{n_rules:,}" if isinstance(n_rules, (int, np.integer)) else "N/A",
        )
    with col3:
        st.metric("Estimated net revenue gain", gain_str)
    with col4:
        st.metric("Targeting advantage vs blanket", adv_str)

    st.divider()

    col_l, col_r = st.columns(2)

    # Top Products
    with col_l:
        st.subheader("Top 10 most ordered products")
        if not data_missing("top_prod", "results/top_products.csv"):
            tp = data["top_prod"].copy()
            if not tp.empty:
                # expected columns: product_name, order_count
                fig = px.bar(
                    tp.head(10).sort_values("order_count"),
                    x="order_count",
                    y="product_name",
                    orientation="h",
                    color="order_count",
                    color_continuous_scale="Blues",
                    labels={"order_count": "Number of orders", "product_name": ""},
                )
                fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Top products table is empty. Check pipeline filters.")

    # Algorithm Comparison
    with col_r:
        st.subheader("Algorithm performance comparison")
        if not data_missing("algo_comp", "results/algorithm_comparison.csv"):
            ac = data["algo_comp"].copy()
            if not ac.empty:
                # expected columns: Algorithm, Itemsets, Time (s)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(
                        x=ac["Algorithm"],
                        y=ac["Itemsets"],
                        name="Itemsets found",
                        marker_color="#2b6cb0",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=ac["Algorithm"],
                        y=ac["Time (s)"],
                        name="Time (s)",
                        mode="lines+markers",
                        line=dict(color="#e53e3e", width=3),
                        marker=dict(size=9),
                    ),
                    secondary_y=True,
                )
                fig.update_layout(height=400, title_text="Itemsets vs execution time")
                fig.update_yaxes(title_text="Itemsets", secondary_y=False)
                fig.update_yaxes(title_text="Time (seconds)", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Algorithm comparison table is empty.")

    st.markdown(
        """
    <div class="insight-box">
    Strong association rules and fast FP‑Growth execution make it feasible to refresh 
    recommendations regularly on a large retail dataset. High‑value bundles can then 
    be fed into pricing and promotion strategies rather than staying at exploratory level.
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE 2: CUSTOMER SEGMENTS
# =============================================================================
elif page == "Customer Segments":
    st.title("Customer Segmentation")
    st.markdown("RFM features combined with K‑Means clustering.")
    st.divider()

    if data_missing("segments", "results/customer_segments.csv"):
        st.stop()

    seg = data["segments"].copy()
    if seg.empty:
        st.info("Customer segments table is empty.")
        st.stop()

    # Expect columns: user_id, frequency, recency, monetary, segment, buyer_type
    seg_summary = (
        seg.groupby("segment")
        .agg(
            Customers=("user_id", "count"),
            Avg_Frequency=("frequency", "mean"),
            Avg_Recency=("recency", "mean"),
            Avg_Monetary=("monetary", "mean"),
        )
        .reset_index()
        .round(1)
    )

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Segment size")
        fig = px.pie(
            seg_summary,
            names="segment",
            values="Customers",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Segment profiles")
        display_df = seg_summary.rename(
            columns={
                "segment": "Segment",
                "Customers": "Number of customers",
                "Avg_Frequency": "Orders per customer",
                "Avg_Recency": "Days between orders",
                "Avg_Monetary": "Total estimated spend",
            }
        )
        st.dataframe(display_df.set_index("Segment"), use_container_width=True)

    st.divider()

    st.subheader("Customer map: frequency vs monetary value")
    if len(seg) > 0:
        sample = seg.sample(min(3000, len(seg)), random_state=42)
        fig = px.scatter(
            sample,
            x="frequency",
            y="monetary",
            color="segment",
            size=None,
            hover_data=["recency", "buyer_type"],
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                "frequency": "Number of orders",
                "monetary": "Estimated total spend",
                "recency": "Average days since prior order",
            },
            opacity=0.65,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of monetary value by segment")
    fig = px.box(
        seg,
        x="segment",
        y="monetary",
        color="segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"monetary": "Estimated total spend", "segment": ""},
        points="outliers",
    )
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
    <div class="insight-box">
    High‑frequency, high‑monetary segments concentrate a large share of revenue. 
    They are natural candidates for value‑adding bundles and loyalty programmes, 
    while low‑frequency segments can be addressed with re‑activation offers.
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE 3: PRODUCT ASSOCIATIONS
# =============================================================================
elif page == "Product Associations":
    st.title("Product Associations & Bundles")
    st.markdown("Association rules derived from FP‑Growth on Instacart baskets.")
    st.divider()

    if data_missing("rules", "results/association_rules.csv"):
        st.stop()

    rules = data["rules"].copy()
    if rules.empty:
        st.info("Association rules table is empty.")
        st.stop()

    # Sliders for rule quality filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider("Minimum support", 0.01, 0.3, 0.03, 0.01)
    with col2:
        min_confidence = st.slider("Minimum confidence", 0.1, 1.0, 0.30, 0.05)
    with col3:
        min_lift = st.slider("Minimum lift", 1.0, 10.0, 1.2, 0.1)

    filtered = rules[
        (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
        & (rules["lift"] >= min_lift)
    ]

    st.markdown(f"{len(filtered)} rules satisfy the selected thresholds.")

    # Top rules table
    st.subheader("Top association rules (by lift)")
    if (
        "antecedents_str" in filtered.columns
        and "consequents_str" in filtered.columns
        and not filtered.empty
    ):
        display_rules = (
            filtered.sort_values("lift", ascending=False)
            .head(20)[
                ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
            ]
            .rename(
                columns={
                    "antecedents_str": "If the customer buys",
                    "consequents_str": "They are likely to also buy",
                    "support": "Support",
                    "confidence": "Confidence",
                    "lift": "Lift",
                }
            )
        )
        display_rules[["Support", "Confidence", "Lift"]] = display_rules[
            ["Support", "Confidence", "Lift"]
        ].round(3)
        st.dataframe(display_rules, use_container_width=True, height=420)
    else:
        st.info("No rules match the current filters.")

    st.divider()

    # Support vs confidence map
    if not filtered.empty:
        st.subheader("Rule map: support vs confidence")
        fig = px.scatter(
            filtered,
            x="support",
            y="confidence",
            color="lift",
            size="lift",
            hover_data=["antecedents_str", "consequents_str"],
            color_continuous_scale="Viridis",
            labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
            opacity=0.75,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # High-utility rules (optional)
    if data["utility"] is not None and not data["utility"].empty:
        st.divider()
        st.subheader("High‑utility rules (focus on high‑value baskets)")
        util_rules = data["utility"].copy()
        util_rules = util_rules.sort_values("utility_score", ascending=False).head(10)
        util_rules[
            ["support", "confidence", "lift", "utility_score"]
        ] = util_rules[["support", "confidence", "lift", "utility_score"]].round(3)
        st.dataframe(
            util_rules.rename(
                columns={
                    "antecedents_str": "If the customer buys",
                    "consequents_str": "They are likely to also buy",
                    "utility_score": "Utility score",
                }
            ),
            use_container_width=True,
        )
        st.markdown(
            """
        <div class="insight-box">
        These rules are mined from high‑value orders only and ranked by a composite 
        utility score (support × confidence × lift). They represent the bundles most 
        relevant for revenue‑focused promotions.
        </div>
        """,
            unsafe_allow_html=True,
        )

# =============================================================================
# PAGE 4: REVENUE SIMULATION
# =============================================================================
elif page == "Revenue Simulation":
    st.title("Revenue Simulation from Product Bundles")
    st.markdown("Estimate the financial impact of deploying association‑rule‑based bundles.")
    st.divider()

    if data_missing("revenue", "results/revenue_simulation.csv"):
        st.stop()

    rev = data["revenue"].copy()
    if rev.empty:
        st.info("Revenue simulation table is empty.")
        st.stop()

    # Interactive parameters
    st.subheader("Simulation parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_customers = st.number_input(
            "Customer base size",
            min_value=1_000,
            max_value=500_000,
            value=50_000,
            step=1_000,
        )
    with col2:
        avg_order_val = st.number_input(
            "Average order value",
            min_value=5.0,
            max_value=200.0,
            value=25.0,
            step=1.0,
        )
    with col3:
        discount_pct = st.slider("Bundle discount (%)", 5, 30, 10)

    # Recalculate metrics using user parameters
    rev_sim = rev.copy()
    rev_sim["est_new_orders"] = (
        rev_sim["support"] * n_customers * rev_sim["confidence"]
    ).astype(int)
    rev_sim["revenue_from_lift"] = (
        rev_sim["est_new_orders"] * avg_order_val * (rev_sim["lift"] - 1)
    )
    rev_sim["discount_cost"] = (
        rev_sim["est_new_orders"] * avg_order_val * (discount_pct / 100)
    )
    rev_sim["net_revenue_gain"] = (
        rev_sim["revenue_from_lift"] - rev_sim["discount_cost"]
    )
    rev_sim["roi_pct"] = (
        (rev_sim["net_revenue_gain"] / (rev_sim["discount_cost"] + 1e-6)) * 100
    ).round(1)
    rev_sim = rev_sim.sort_values("net_revenue_gain", ascending=False)

    total_gain = rev_sim["net_revenue_gain"].sum()
    total_cost = rev_sim["discount_cost"].sum()
    total_lift_rev = rev_sim["revenue_from_lift"].sum()
    avg_roi = rev_sim["roi_pct"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total revenue uplift", f"${total_lift_rev:,.0f}")
    col2.metric("Total discount cost", f"${total_cost:,.0f}")
    col3.metric("Net incremental revenue", f"${total_gain:,.0f}")
    col4.metric("Average ROI on bundles", f"{avg_roi:.0f}%")

    st.divider()

    st.subheader("Top bundles by net revenue gain")
    top15 = rev_sim.head(15).copy()
    top15["bundle"] = (
        top15["antecedents_str"] + " → " + top15["consequents_str"]
    ).str.slice(0, 60)

    fig = px.bar(
        top15.sort_values("net_revenue_gain"),
        x="net_revenue_gain",
        y="bundle",
        orientation="h",
        color="roi_pct",
        color_continuous_scale="RdYlGn",
        labels={
            "net_revenue_gain": "Net revenue gain",
            "bundle": "",
            "roi_pct": "ROI (%)",
        },
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue uplift vs discount cost")
    fig = px.scatter(
        rev_sim.head(100),
        x="discount_cost",
        y="revenue_from_lift",
        size="est_new_orders",
        color="roi_pct",
        hover_data=["antecedents_str", "consequents_str"],
        color_continuous_scale="RdYlGn",
        labels={
            "discount_cost": "Discount cost",
            "revenue_from_lift": "Revenue uplift",
        },
    )
    max_val = float(
        max(rev_sim["discount_cost"].max(), rev_sim["revenue_from_lift"].max())
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Break‑even",
        )
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detailed table"):
        st.dataframe(
            rev_sim[
                [
                    "antecedents_str",
                    "consequents_str",
                    "est_new_orders",
                    "revenue_from_lift",
                    "discount_cost",
                    "net_revenue_gain",
                    "roi_pct",
                ]
            ]
            .rename(
                columns={
                    "antecedents_str": "If buys",
                    "consequents_str": "Also buys",
                    "est_new_orders": "Estimated orders",
                    "revenue_from_lift": "Revenue uplift",
                    "discount_cost": "Discount cost",
                    "net_revenue_gain": "Net gain",
                    "roi_pct": "ROI (%)",
                }
            )
            .round(2),
            use_container_width=True,
        )

# =============================================================================
# PAGE 5: PROMOTION ROI
# =============================================================================
elif page == "Promotion ROI":
    st.title("Promotion Efficiency")
    st.markdown(
        "Comparison between targeted, data‑driven discounts and non‑targeted blanket discounts."
    )
    st.divider()

    if data_missing("promotion", "results/promotion_efficiency.csv"):
        st.stop()

    promo = data["promotion"].copy()
    if promo.empty:
        st.info("Promotion efficiency table is empty.")
        st.stop()

    st.subheader("Promotion parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        blanket_disc = st.slider("Blanket discount (%)", 5, 30, 15)
    with col2:
        targeted_disc = st.slider("Targeted discount (%)", 5, 20, 10)
    with col3:
        targeted_lift = st.slider(
            "Expected lift from targeting", 1.05, 2.0, 1.25, 0.05
        )

    # Base columns from CSV: blanket_cost, targeted_cost, targeted_gain, advantage
    # Use them as default scenario, then scale relative to sliders if desired.
    scale_blanket = blanket_disc / 15  # 15 is the baseline used in your CSV
    scale_targeted = targeted_disc / 10  # 10 is typical targeted baseline
    scale_lift = targeted_lift / 1.25

    promo["blanket_net_gain"] = -(promo["blanket_cost"] * scale_blanket)
    promo["targeted_revenue_lift"] = promo["targeted_gain"] * scale_lift
    promo["targeted_discount_cost"] = promo["targeted_cost"] * scale_targeted
    promo["targeted_net_gain"] = (
        promo["targeted_revenue_lift"] - promo["targeted_discount_cost"]
    )
    promo["targeting_advantage"] = promo["targeted_net_gain"] - promo["blanket_net_gain"]

    total_targeted = promo["targeted_net_gain"].sum()
    total_blanket = promo["blanket_net_gain"].sum()
    advantage = promo["targeting_advantage"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Blanket discount P&L", f"${total_blanket:,.0f}")
    col2.metric(
        "Targeted discount P&L",
        f"${total_targeted:,.0f}",
        delta=f"${total_targeted - total_blanket:,.0f} vs blanket",
    )
    col3.metric("Targeting advantage", f"${advantage:,.0f}")

    st.divider()

    st.subheader("P&L by customer segment")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Blanket discount P&L",
            x=promo["segment"],
            y=promo["blanket_net_gain"],
            marker_color="#e53e3e",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Targeted discount P&L",
            x=promo["segment"],
            y=promo["targeted_net_gain"],
            marker_color="#2f855a",
        )
    )
    fig.update_layout(
        barmode="group",
        height=420,
        yaxis_title="Net P&L",
        xaxis_title="Customer segment",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost vs revenue lift by segment")
    melted = promo.melt(
        id_vars="segment",
        value_vars=["targeted_discount_cost", "targeted_revenue_lift"],
        var_name="Component",
        value_name="Amount",
    )
    fig = px.bar(
        melted,
        x="segment",
        y="Amount",
        color="Component",
        barmode="group",
        color_discrete_map={
            "targeted_discount_cost": "#e53e3e",
            "targeted_revenue_lift": "#2f855a",
        },
        labels={"segment": "Customer segment", "Amount": "Amount"},
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
    <div class="insight-box">
    Blanket discounts reduce margin for all customers, regardless of their likelihood 
    to respond. Targeted promotions focus investment where lift is highest, turning 
    a cost centre into a controlled growth lever.
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Detailed segment table"):
        display_promo = promo.round(0)[
            [
                "segment",
                "customers",
                "avg_spend",
                "blanket_net_gain",
                "targeted_discount_cost",
                "targeted_revenue_lift",
                "targeted_net_gain",
                "targeting_advantage",
            ]
        ]
        display_promo.columns = [
            "Segment",
            "Customers",
            "Average spend",
            "Blanket P&L",
            "Targeted discount cost",
            "Targeted revenue lift",
            "Targeted net",
            "Advantage vs blanket",
        ]
        st.dataframe(display_promo, use_container_width=True)
