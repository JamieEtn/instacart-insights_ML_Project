# =============================================================================
# BUSINESS ANALYSIS – RETAIL INSIGHTS
# Faithful to the DSTI report — no additional analyses
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import warnings, os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
           "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B4FDB"]
os.makedirs('report_imgs', exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================
df       = pd.read_csv('data/processed/master.csv')
segments = pd.read_csv('results/customer_segments.csv')
rules    = pd.read_csv('results/association_rules.csv')
revenue  = pd.read_csv('results/revenue_simulation.csv')

# normalize column names from pipeline; older outputs may lack the derived fields
if 'total_spend' not in segments.columns and 'monetary' in segments.columns:
    segments['total_spend'] = segments['monetary']

# compute average basket size per user if not present
if 'avg_basket_size' not in segments.columns:
    basket = df.groupby('user_id').size().rename('avg_basket_size')
    segments = segments.merge(basket, on='user_id', how='left')

basket_size     = df.groupby('order_id').size()
top_products    = df['product_name'].value_counts().head(10)
hourly_orders   = df.groupby('order_hour_of_day')['order_id'].nunique()
segment_summary = segments.groupby('segment').agg(
    frequency=('frequency','mean'),
    total_spend=('total_spend','mean'),
    avg_basket_size=('avg_basket_size','mean')
).round(2)
top_rules    = rules.sort_values('lift', ascending=False).head(10)
top_revenue  = revenue.sort_values('net_revenue_gain', ascending=False).head(10)
best_segment = segment_summary['total_spend'].idxmax()
best_rule    = top_revenue.iloc[0]
peak_hour    = hourly_orders.idxmax()

# Promotion efficiency (from the report)
promo_data = {
    'segment': ['Seg. 0', 'Seg. 1', 'Seg. 2 ★', 'Seg. 3'],
    'profile': ['Mid-Range', 'Small Basket', 'Heavy Buyer', 'Occasional'],
    'avg_spend': [71.33, 26.55, 130.15, 26.41],
    'blanket_15': [-10.70, -3.98, -19.52, -3.96],
    'targeted_10': [17.83, 6.64, 32.54, 6.60],
    'advantage': [7.14, 2.65, 13.01, 2.64]
}
promo_df = pd.DataFrame(promo_data)

def save_fig(name):
    path = f'report_imgs/{name}.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    return path

def img_block(path, width=16*cm):
    return Image(path, width=width, height=width*0.48)

print("✅ Data loaded")

# =============================================================================
# G1 — Orders by hour
# =============================================================================
fig, ax = plt.subplots(figsize=(10,4))
ax.fill_between(hourly_orders.index, hourly_orders.values, alpha=0.25, color=PALETTE[0])
ax.plot(hourly_orders.index, hourly_orders.values, color=PALETTE[0], linewidth=2.5)
ax.axvline(peak_hour, color=PALETTE[2], linestyle='--', label=f'Peak: {peak_hour}h', linewidth=1.8)
ax.set_title("Orders by Hour of Day", fontsize=14, fontweight='bold')
ax.set_xlabel("Hour"); ax.set_ylabel("Number of Orders"); ax.legend()
img_hourly = save_fig('hourly_orders')

# =============================================================================
# G2 — Top 10 products
# =============================================================================
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(x=top_products.values, y=top_products.index, palette=PALETTE, ax=ax)
ax.set_title("Top 10 Best-Selling Products", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of Sales")
for i, v in enumerate(top_products.values):
    ax.text(v+50, i, f'{v:,}', va='center', fontsize=9)
img_top_products = save_fig('top_products')

# =============================================================================
# G3 — Segmentation: comparison of the 4 segments
# =============================================================================
fig, axes = plt.subplots(1,3, figsize=(14,4))
metrics = ['frequency','total_spend','avg_basket_size']
titles  = ['Avg. Frequency','Avg. Spend ($)','Avg. Basket Size']
profiles = ['Mid-Range\nBuyers','Small\nBaskets','★ Heavy\nBuyers','Occasional\nShoppers']

for i, (m, t) in enumerate(zip(metrics, titles)):
    bars = axes[i].bar(profiles, segment_summary[m], color=PALETTE[:4])
    axes[i].set_title(t, fontweight='bold')
    for bar, val in zip(bars, segment_summary[m]):
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    best_idx = list(segment_summary.index).index(best_segment)
    bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(3)

plt.suptitle("KMeans Segmentation — 4 Customer Profiles  ⭐ = Priority Segment",
             fontsize=13, fontweight='bold')
img_segments = save_fig('segment_comparison')

# =============================================================================
# G4 — RFM: Champions vs Regular breakdown
# =============================================================================
fig, axes = plt.subplots(1,2, figsize=(12,5))

# RFM Pie
rfm_labels  = ['Champions\n(50%)', 'Regular Buyers\n(50%)']
rfm_values  = [50, 50]
rfm_colors  = [PALETTE[2], PALETTE[0]]
axes[0].pie(rfm_values, labels=rfm_labels, colors=rfm_colors,
            autopct='%1.0f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':3},
            textprops={'fontsize':11})
axes[0].set_title("RFM Breakdown — Champions vs Regular", fontweight='bold', fontsize=12)

# Spend by segment (horizontal bar)
seg_labels = ['Seg. 0\nMid-Range', 'Seg. 1\nSmall Basket',
              'Seg. 2 ★\nHeavy Buyer', 'Seg. 3\nOccasional']
seg_spend  = [71.33, 26.55, 130.15, 26.41]
bar_colors = [PALETTE[0], PALETTE[5], PALETTE[2], PALETTE[4]]
bars = axes[1].barh(seg_labels, seg_spend, color=bar_colors)
axes[1].set_title("Average Spend by Segment ($)", fontweight='bold', fontsize=12)
axes[1].set_xlabel("Average Spend ($)")
bars[2].set_edgecolor('gold'); bars[2].set_linewidth(3)
for bar, val in zip(bars, seg_spend):
    axes[1].text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                 f'${val:.2f}', va='center', fontweight='bold', fontsize=10)

plt.suptitle("RFM Analysis & Segment Value", fontsize=14, fontweight='bold')
img_rfm = save_fig('rfm_segments')

# =============================================================================
# G5 — Association rules: scatter lift + barplot
# =============================================================================
fig, axes = plt.subplots(1,2, figsize=(14,5))

sc = axes[0].scatter(top_rules['support'], top_rules['confidence'],
                     s=top_rules['lift']*60, c=top_rules['lift'],
                     cmap='YlOrRd', alpha=0.85, edgecolors='gray')
plt.colorbar(sc, ax=axes[0], label='Lift')
axes[0].set_title("Association Rules\n(size & color = lift)", fontweight='bold')
axes[0].set_xlabel("Support"); axes[0].set_ylabel("Confidence")
axes[0].axhline(0.30, color='blue', linestyle=':', alpha=0.6, label='min confidence 30%')
axes[0].axvline(0.03, color='green', linestyle=':', alpha=0.6, label='min support 3%')
axes[0].legend(fontsize=8)

short_labels = [f"{r['antecedents_str'][:22]}→{r['consequents_str'][:15]}"
                for _,r in top_rules.iterrows()]
bar_colors_lift = [PALETTE[2] if l >= 2.0 else PALETTE[0] if l >= 1.5 else PALETTE[5]
                   for l in top_rules['lift'].values]
axes[1].barh(short_labels[::-1], top_rules['lift'].values[::-1], color=bar_colors_lift[::-1])
axes[1].axvline(1.5, color='red', linestyle='--', alpha=0.7, label='Actionable threshold (1.5)')
axes[1].axvline(1.0, color='gray', linestyle=':', alpha=0.5)
axes[1].set_title("Top 10 Rules by Lift", fontweight='bold')
axes[1].set_xlabel("Lift"); axes[1].legend(fontsize=8)

plt.suptitle("FP-Growth — Product Association Rules", fontsize=14, fontweight='bold')
img_rules = save_fig('association_rules')

# =============================================================================
# G6 — Association heatmap
# =============================================================================
pivot_rules = top_rules.pivot_table(
    index='antecedents_str', columns='consequents_str',
    values='lift', aggfunc='max').fillna(0)
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(pivot_rules, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label':'Lift'})
ax.set_title("Product Association Heatmap (Lift)", fontsize=13, fontweight='bold')
plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0)
img_heatmap = save_fig('heatmap_rules')

# =============================================================================
# G7 — Simulated revenues
# =============================================================================
fig, ax = plt.subplots(figsize=(10,5))
short_rev = [f"{r['antecedents_str'][:22]}→{r['consequents_str'][:15]}"
             for _,r in top_revenue.iterrows()]
bar_rev = ax.barh(short_rev[::-1], top_revenue['net_revenue_gain'].values[::-1]/1000,
                  color=sns.color_palette("YlGn_r", len(top_revenue)))
for bar, val in zip(bar_rev, top_revenue['net_revenue_gain'].values[::-1]/1000):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f'${val:.1f}K', va='center', fontsize=8, fontweight='bold')
ax.set_title("Top 10 Revenue Opportunities — Estimated Net Gains",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Estimated Net Gain ($K)")
ax.axvline(top_revenue['net_revenue_gain'].mean()/1000, color='red',
           linestyle='--', alpha=0.6, label='Average')
ax.legend()
img_revenue = save_fig('revenue_opportunities')

# =============================================================================
# G8 — Targeted vs Blanket promotions
# =============================================================================
fig, axes = plt.subplots(1,2, figsize=(13,5))

x = np.arange(len(promo_df))
w = 0.35
b1 = axes[0].bar(x - w/2, promo_df['blanket_15'], w,
                 label='Blanket 15%', color=PALETTE[3], alpha=0.85)
b2 = axes[0].bar(x + w/2, promo_df['targeted_10'], w,
                 label='Targeted 10%', color=PALETTE[5], alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(promo_df['segment'])
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_title("Blanket 15% vs Targeted 10%\n(cost/gain per customer per campaign $)",
                  fontweight='bold')
axes[0].set_ylabel("$ per customer"); axes[0].legend()
for bar in b1:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()-1.5,
                 f'${bar.get_height():.1f}', ha='center', fontsize=8, color='white', fontweight='bold')
for bar in b2:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'+${bar.get_height():.1f}', ha='center', fontsize=8, fontweight='bold')

axes[1].bar(promo_df['segment'], promo_df['advantage'],
            color=[PALETTE[2] if s == 'Seg. 2 ★' else PALETTE[0] for s in promo_df['segment']])
axes[1].set_title("Targeted vs Blanket Advantage\n($ per customer — Targeted always wins)",
                  fontweight='bold')
axes[1].set_ylabel("Advantage ($)")
for i, val in enumerate(promo_df['advantage']):
    axes[1].text(i, val+0.1, f'+${val:.2f}', ha='center', fontweight='bold', fontsize=10)

plt.suptitle("Promotion Efficiency — Targeted vs Blanket", fontsize=14, fontweight='bold')
img_promo = save_fig('promotion_efficiency')

print("✅ All charts generated")

# =============================================================================
# PDF GENERATION
# =============================================================================
print("\n📄 Generating PDF report...")

doc = SimpleDocTemplate('retail_insights_report_DSTI.pdf', pagesize=A4,
                        topMargin=1.8*cm, bottomMargin=1.8*cm,
                        leftMargin=2*cm, rightMargin=2*cm)
styles = getSampleStyleSheet()

s_title  = ParagraphStyle('T',  parent=styles['Title'],
                           fontSize=20, spaceAfter=4, textColor=colors.HexColor('#2E86AB'))
s_inst   = ParagraphStyle('I',  parent=styles['Normal'],
                           fontSize=10, spaceAfter=6, textColor=colors.HexColor('#555555'))
s_h1     = ParagraphStyle('H1', parent=styles['Heading1'],
                           fontSize=13, spaceBefore=12, spaceAfter=5,
                           textColor=colors.HexColor('#2E86AB'))
s_h2     = ParagraphStyle('H2', parent=styles['Heading2'],
                           fontSize=10, spaceBefore=6, spaceAfter=3,
                           textColor=colors.HexColor('#A23B72'))
s_body   = ParagraphStyle('B',  parent=styles['Normal'],
                           fontSize=9, spaceAfter=5, leading=13)
s_ok     = ParagraphStyle('OK', parent=styles['Normal'],
                           fontSize=9, spaceAfter=5, leading=13,
                           backColor=colors.HexColor('#E8F8F0'),
                           borderPadding=6, leftIndent=8, rightIndent=8)
s_warn   = ParagraphStyle('W',  parent=styles['Normal'],
                           fontSize=9, spaceAfter=5, leading=13,
                           backColor=colors.HexColor('#FEF3E0'),
                           borderPadding=6, leftIndent=8, rightIndent=8)
s_note   = ParagraphStyle('N',  parent=styles['Normal'],
                           fontSize=8, spaceAfter=4, leading=11,
                           textColor=colors.HexColor('#777777'))

def make_table(data, col_widths, header_color='#2E86AB'):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',     (0,0),(-1,0), colors.HexColor(header_color)),
        ('TEXTCOLOR',      (0,0),(-1,0), colors.white),
        ('FONTNAME',       (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0),(-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.HexColor('#F5F9FF'), colors.white]),
        ('GRID',           (0,0),(-1,-1), 0.4, colors.grey),
        ('PADDING',        (0,0),(-1,-1), 5),
        ('ALIGN',          (1,1),(-1,-1), 'CENTER'),
        ('VALIGN',         (0,0),(-1,-1), 'MIDDLE'),
    ]))
    return t

story = []

# ── COVER PAGE ─────────────────────────────────────────────────────────────────
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph("DATA SCIENCE TECH INSTITUTE", s_inst))
story.append(Paragraph("Applied MSc in Data Science &amp; AI", s_inst))
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("RETAIL INSIGHTS — Business Analysis Report", s_title))
story.append(Paragraph(
    "Customer Segmentation · Product Bundles · Revenue Simulation · Promotion ROI", s_inst))
story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2E86AB')))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "Dataset: Instacart Online Grocery — 3M+ transactions · 200,000+ customers · "
    "Pipeline: FP-Growth Association Rules · RFM Analysis · KMeans Clustering", s_note))
story.append(Paragraph(
    "<i>Data Transparency Note: Monetary figures use simulated department-level prices "
    "(uniform random, seed=42). All financial figures are indicative — for strategic framing only.</i>",
    s_note))
story.append(Spacer(1, 0.5*cm))

# Key KPIs
kpi_data = [
    ['KPI', 'Value', 'Source'],
    ['Average Basket Size',          '10.57 items',      'Measured'],
    ['Avg. Reorder Rate',            '45.1%',            'Measured'],
    ['Customer Segments',            '4 (KMeans)',       'Model'],
    ['Max Lift (Limes→Lemon)',       '2.38x',            'FP-Growth'],
    ['Actionable Rules ≥1.5 Lift',   '64.5% (20/31)',    'FP-Growth'],
    ['Top 10 Bundle Gain',           '$482,976',         'Simulated*'],
    ['Targeted vs Blanket Advantage','$63,600 / 10K cust.','Simulated*'],
]
story.append(make_table(kpi_data, [6.5*cm, 4.5*cm, 3.5*cm]))
story.append(PageBreak())

# ── EXECUTIVE SUMMARY ──────────────────────────────────────────────────────────
story.append(Paragraph("Executive Summary", s_h1))
story.append(Paragraph(
    "This report translates the outputs of a machine learning pipeline applied to over 3 million "
    "transactions into actionable business recommendations. Three analytical modules were deployed: "
    "<b>FP-Growth</b> to identify product co-purchases, <b>RFM + KMeans</b> to segment the customer base, "
    "and a <b>revenue simulation model</b> to quantify the financial impact of targeted promotions.",
    s_body))
story.append(Paragraph(
    "The central conclusion is clear: <b>replacing blanket discounts with targeted promotions "
    "generates more revenue at a lower promotional cost.</b> Shifting from a 15% blanket discount "
    "to a targeted 10% promotion generates an advantage of up to <b>+$13 per customer per campaign</b> "
    "for the highest-value segment. The top 10 bundle opportunities represent "
    "<b>$482,000+ in estimated incremental net revenue</b>.", s_body))
story.append(Spacer(1, 0.3*cm))
story.append(PageBreak())

# ── SECTION 1: CUSTOMER BEHAVIOR ───────────────────────────────────────────────
story.append(Paragraph("1. Customer Behavior & Top Products", s_h1))
story.append(Paragraph(
    f"The average basket is <b>10.57 items</b> (median: 9). Peak activity is at "
    f"<b>{peak_hour}:00</b> — the optimal window for push notifications and flash promotions. "
    f"Bananas (conventional and organic) dominate sales, followed by Organic Strawberries "
    f"and Spinach — these staple products are natural levers for triggering cross-purchases.",
    s_body))
story.append(img_block(img_hourly))
story.append(img_block(img_top_products))
story.append(Paragraph("💡 Operational Insight", s_h2))
story.append(Paragraph(
    f"Scheduling push campaigns precisely at {peak_hour}:00 maximizes conversion rates. "
    f"Top 10 products must have zero stockouts — unavailability triggers immediate dissatisfaction "
    f"among Heavy Buyers (Seg. 2), who account for 50% of total value.",
    s_ok))
story.append(PageBreak())

# ── SECTION 2: SEGMENTATION & RFM ─────────────────────────────────────────────
story.append(Paragraph("2. Customer Segmentation & RFM Analysis", s_h1))
story.append(Paragraph(
    "KMeans clustering (k=4) combined with RFM scoring identifies four distinct behavioral profiles. "
    "This understanding enables precise promotional targeting rather than costly universal discounts.",
    s_body))

story.append(Paragraph("2.1 The Four Segments", s_h2))
seg_table_data = [
    ['Segment', 'Profile', 'Avg. Spend*', 'Basket Size', 'Variability', 'Priority'],
    ['Seg. 0',   'Mid-Range Buyers',       '$71.33',  '17.3 items', 'Medium', 'Upsell'],
    ['Seg. 1',   'Small Baskets',          '$26.55',  '6.5 items',  'Low',    'Activation'],
    ['Seg. 2 ★', 'Heavy Buyers — TOP',     '$130.15', '30.5 items', 'High',   'RETENTION'],
    ['Seg. 3',   'Occasional Shoppers',    '$26.41',  '6.6 items',  'Low',    'Re-engage'],
]
t = make_table(seg_table_data, [2.5*cm, 3.5*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm], '#A23B72')
t.setStyle(TableStyle([
    ('BACKGROUND',  (0,0),(-1,0), colors.HexColor('#A23B72')),
    ('TEXTCOLOR',   (0,0),(-1,0), colors.white),
    ('FONTNAME',    (0,0),(-1,0), 'Helvetica-Bold'),
    ('BACKGROUND',  (0,3),(-1,3), colors.HexColor('#FFF3CD')),
    ('FONTNAME',    (0,3),(-1,3), 'Helvetica-Bold'),
    ('FONTSIZE',    (0,0),(-1,-1), 8),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F9EEF5'), colors.white]),
    ('GRID',        (0,0),(-1,-1), 0.4, colors.grey),
    ('PADDING',     (0,0),(-1,-1), 5),
    ('ALIGN',       (1,1),(-1,-1), 'CENTER'),
]))
story.append(t)
story.append(Paragraph("* Spend modeled using simulated prices (no real prices in the Instacart dataset).", s_note))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("2.2 Basket Variability", s_h2))
story.append(Paragraph(
    "Segment 2 averages 30.5 items — <b>almost 5x the size</b> of Segments 1 and 3 (~6.5 items). "
    "This is not just a size difference: Heavy Buyers show a higher standard deviation in basket "
    "composition, meaning they explore the catalog more broadly. This is the segment most receptive "
    "to new product suggestions and premium bundles.", s_body))
story.append(img_block(img_segments))
story.append(img_block(img_rfm))

story.append(Paragraph("2.3 RFM Behavioral Profiles", s_h2))
story.append(Paragraph(
    "<b>50%</b> of the customer base qualifies as 'Champions' (high frequency + high monetary value) "
    "and <b>50%</b> as 'Regular Buyers'. This 50/50 split is strategically important: "
    "Champions generate a disproportionate share of revenue at a relatively low acquisition cost "
    "— they are already engaged. <b>Protecting this group through VIP treatment is the highest-ROI "
    "initiative available.</b>", s_body))

rfm_table = [
    ['RFM Profile', 'Share', 'Behavior', 'Recommended Action'],
    ['Champions',      '50%', 'High freq + High spend', 'VIP Program: cashback, early access, priority service'],
    ['Regular Buyers', '50%', 'High freq + Lower spend', 'Upsell via premium bundles, organic cross-sell'],
    ['Big Baskets*',   '—',   'Low freq + High spend',  'Increase frequency: automated reorder reminders'],
    ['Dormants*',      '—',   'Low freq + Low spend',   'Reactivation campaign: strong -15% coupon'],
]
story.append(make_table(rfm_table, [3*cm, 1.5*cm, 4.5*cm, 7.5*cm], '#2E86AB'))
story.append(Paragraph("* Sub-profiles of the Regular Buyers group based on the Value × Frequency BCG matrix.", s_note))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Business Implication — Segmentation", s_h2))
story.append(Paragraph(
    "Allocate at least <b>50% of the promotional budget to Segment 2</b> (Heavy Buyers, ~$130 spend). "
    "Losing even a fraction of this segment has a disproportionate revenue impact. For Segments 1 &amp; 3 "
    "(~$26 spend), use low-cost discovery offers (e.g. 5% on first bundle) to gradually grow basket size. "
    "<b>DO NOT apply uniform discounts across all segments</b> — the data shows this wastes budget "
    "on customers who would have purchased anyway.", s_warn))
story.append(PageBreak())

# ── SECTION 3: ASSOCIATION RULES ───────────────────────────────────────────────
story.append(Paragraph("3. Product Association Rules & Bundle Strategy", s_h1))
story.append(Paragraph(
    "FP-Growth was run on the top 500 most-purchased products by the 5,000 most active customers. "
    "Parameters: <b>min_support=3%</b>, <b>min_confidence=30%</b>, <b>min_lift=1.2</b>. "
    "Of 31 rules generated, <b>20 (64.5%) have a lift ≥ 1.5</b> — the threshold above which a bundle "
    "promotion is commercially viable. Lift measures how many times more likely a joint purchase of two "
    "products is compared to chance: <b>lift=2.38 means 2.38x more likely than random.</b>",
    s_body))

story.append(Paragraph("3.1 Top Bundle Opportunities", s_h2))
bundle_table = [
    ['If the customer buys…', 'Suggest…', 'Support', 'Confidence', 'Lift', 'Type'],
    ['Limes',                       'Large Lemon',          '5.31%', '39.4%', '2.38 ★★', 'Citrus Bundle'],
    ['Large Lemon',                  'Limes',                '5.31%', '32.0%', '2.38 ★★', 'Citrus Bundle'],
    ['Org. Hass Avoc. + Org. Straw.','Bag Org. Bananas',    '3.01%', '54.7%', '2.19',    'Brunch Bio Pack'],
    ['Bag Org. Bananas + Org. Straw.','Org. Hass Avocado',  '3.01%', '33.9%', '2.12',    'Brunch Bio Pack'],
    ['Organic Hass Avocado',         'Bag Organic Bananas',  '7.41%', '46.4%', '1.86',    'Endcap Display'],
    ['Organic Lemon',                'Bag Organic Bananas',  '3.53%', '46.3%', '1.85',    'Organic Bundle'],
    ['Organic Avocado',              'Large Lemon',          '4.19%', '30.4%', '1.84',    'Citrus Bundle'],
    ['Organic Raspberries',          'Organic Strawberries', '5.79%', '40.2%', '1.77',    'Red Fruit Pack'],
]
story.append(make_table(bundle_table,
    [4*cm, 3.5*cm, 1.8*cm, 2*cm, 1.8*cm, 3.4*cm], '#F18F01'))
story.append(Paragraph(
    "Support = % of orders containing both products. Confidence = % of orders containing the antecedent "
    "that also contain the consequent. ★★ = highest lift in the dataset.", s_note))
story.append(Spacer(1, 0.4*cm))

story.append(img_block(img_rules))
story.append(img_block(img_heatmap))

story.append(Paragraph("3.2 How to Activate the Bundles", s_h2))
activation_table = [
    ['Channel', 'Implementation', 'Cost', 'Expected Impact'],
    ['Shelf Placement',   'Limes &amp; Large Lemon in adjacent zones',              '~$0',         'Immediate'],
    ['Digital Cross-sell','\'Frequently bought together\' widget at checkout',       'Dev time',    '+8-12% basket'],
    ['Pre-made Packs',    'Shrink-wrap Citrus Bundle with -5 to 8% tag',            'Packaging',   '+15% attach rate'],
    ['Endcap Display',    'Monthly rotation of top 3 bundles',                      'Display cost', '+10% visibility'],
]
story.append(make_table(activation_table, [3.5*cm, 5.5*cm, 2.5*cm, 3*cm], '#2E86AB'))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Key Insight — Bundles", s_h2))
story.append(Paragraph(
    "The <b>Citrus Bundle (Limes + Large Lemon, lift=2.38)</b> is the strongest association in the dataset. "
    "With 5.31% support, it already appears in <b>1 in every 20 orders</b> — the opportunity is to "
    "convert the remaining 61% who buy limes WITHOUT buying lemons. "
    "With 39.4% confidence, <b>almost 4 in 10 lime buyers</b> would also buy lemons if prompted. "
    "This is a zero-cost, zero-risk quick win.", s_ok))
story.append(PageBreak())

# ── SECTION 4: REVENUE ─────────────────────────────────────────────────────────
story.append(Paragraph("4. Revenue Simulation", s_h1))
story.append(Paragraph(
    "The revenue model estimates the net financial gain from implementing each association rule "
    "as a targeted promotion. The formula accounts for incremental orders triggered by lift, "
    "average order value, and the cost of a 10% promotional discount.", s_body))

story.append(Paragraph("4.1 Revenue Model Formula", s_h2))
formula_table = [
    ['Step', 'Formula', 'Logic'],
    ['1. New Orders',    'support × N_customers × confidence', 'Orders triggered by the association'],
    ['2. Lift Revenue',  'new_orders × avg_order_value × (lift - 1)', 'Incremental revenue above baseline'],
    ['3. Discount Cost', 'new_orders × avg_order_value × 10%', 'Promotional cost to trigger the bundle'],
    ['4. Net Gain',      'Lift Revenue - Discount Cost', 'What the store actually keeps'],
]
story.append(make_table(formula_table, [3.5*cm, 6*cm, 7*cm], '#393E41'))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("4.2 Top 10 Revenue Opportunities", s_h2))
rev_table = [
    ['Rank', 'Rule', 'Net Gain*', '% of Total'],
    ['#1',  'Limes → Large Lemon',                          '$71,507',  '14.8%'],
    ['#2',  'Org. Hass Avocado → Bag Org. Bananas',         '$69,895',  '14.5%'],
    ['#3',  'Large Lemon → Limes',                          '$58,207',  '12.1%'],
    ['#4',  'Org. Hass Avoc. + Org. Straw. → Bag Org. Ban.','$48,092',  '10.0%'],
    ['#5',  'Org. Raspberries → Bag Org. Bananas',          '$46,273',  '9.6%'],
    ['#6',  'Org. Strawberries → Bag Org. Bananas',         '$43,030',  '8.9%'],
    ['#7',  'Org. Raspberries → Org. Strawberries',         '$42,000',  '8.7%'],
    ['#8',  'Bag Org. Bananas → Org. Strawberries',         '$39,025',  '8.1%'],
    ['#9',  'Org. Lemon → Bag Org. Bananas',                '$32,990',  '6.8%'],
    ['#10', 'Org. Cucumber → Bag Org. Bananas',             '$31,957',  '6.6%'],
    ['TOP 10 TOTAL', '',                                    '$482,976', '100%'],
]
t_rev = make_table(rev_table, [1.5*cm, 8*cm, 3*cm, 2.5*cm], '#C73E1D')
story.append(t_rev)
story.append(Paragraph("* Figures modeled with simulated prices. Relative rankings are reliable; "
                        "absolute values require real prices for confirmation.", s_note))
story.append(Spacer(1, 0.4*cm))
story.append(img_block(img_revenue))

story.append(Paragraph("Business Implication — Revenue", s_h2))
story.append(Paragraph(
    "The <b>top 3 rules alone (Limes/Lemon + Avocado/Bananas)</b> account for 41.4% of the total "
    "estimated gain ($199,609). Launching only these three bundles in Month 1 — with a 5% discount "
    "exclusively for Segment 2 (Heavy Buyers) — maximizes margin by targeting customers with the "
    "highest response probability AND the highest baseline spend.", s_ok))
story.append(PageBreak())

# ── SECTION 5: PROMOTION EFFICIENCY ───────────────────────────────────────────
story.append(Paragraph("5. Promotion Efficiency — Targeted vs Blanket", s_h1))
story.append(Paragraph(
    "This is the central financial question of the report: if I adopt these insights, how much money "
    "do I save — and why? A blanket discount applies the same 15% offer to all customers regardless "
    "of their response probability. A targeted promotion applies a lower 10% discount only to the "
    "segments most likely to increase their basket size, generating a modeled basket lift of +25% "
    "for those segments.", s_body))

story.append(Paragraph("5.1 Per-Customer Per-Campaign Comparison", s_h2))
promo_comp_table = [
    ['Segment', 'Profile', 'Avg. Spend*', 'Blanket 15%\n(cost/customer)', 'Targeted 10%\n(gain/customer)', 'Advantage\n/customer', 'Winner'],
    ['Seg. 0', 'Mid-Range',    '$71.33',  '-$10.70', '+$17.83', '+$7.14',  'Targeted ✓'],
    ['Seg. 1', 'Small Basket', '$26.55',  '-$3.98',  '+$6.64',  '+$2.65',  'Targeted ✓'],
    ['Seg. 2 ★','Heavy Buyer', '$130.15', '-$19.52', '+$32.54', '+$13.01', 'Targeted ✓✓'],
    ['Seg. 3', 'Occasional',   '$26.41',  '-$3.96',  '+$6.60',  '+$2.64',  'Targeted ✓'],
]
story.append(make_table(promo_comp_table, [2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm, 2.5*cm], '#393E41'))
story.append(Spacer(1, 0.4*cm))
story.append(img_block(img_promo))

story.append(Paragraph("5.2 Why Targeted Always Wins", s_h2))
story.append(Paragraph(
    "The blanket discount is a <b>pure cost</b>: you pay 15% on every purchase that would have "
    "happened anyway. The targeted approach flips the logic — you pay 10% only where it triggers "
    "additional purchasing behavior (the lift). Because the modeled basket lift is 25% and you only "
    "pay 10%, you generate a net positive value on every targeted transaction.", s_body))

story.append(Paragraph("5.3 Projected Savings at Scale (10,000 customers)", s_h2))
scale_table = [
    ['Segment', 'Customers', 'Advantage/customer', 'Total Campaign Gain'],
    ['Seg. 0',  '~2,500',  '+$7.14',  '+$17,850'],
    ['Seg. 1',  '~2,500',  '+$2.65',  '+$6,625'],
    ['Seg. 2 ★','~2,500',  '+$13.01', '+$32,525'],
    ['Seg. 3',  '~2,500',  '+$2.64',  '+$6,600'],
    ['TOTAL',   '10,000',  'avg. +$6.36', '+$63,600 per campaign'],
]
story.append(make_table(scale_table, [3*cm, 3*cm, 3.5*cm, 7*cm], '#2E86AB'))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Conclusion — How Much Do You Save and Why?", s_h2))
story.append(Paragraph(
    "Switching from blanket promotions to targeted promotions saves approximately "
    "<b>$63,600 per campaign of 10,000 customers</b> — that is money previously given away as "
    "unnecessary discounts to customers who would have purchased regardless. "
    "The mechanism is twofold: (1) you pay a lower discount rate (10% vs 15%), "
    "and (2) you only discount where it generates incremental volume (lift &gt; 1). "
    "Over <b>12 monthly campaigns</b>, this structural improvement is worth "
    "<b>~$763,000/year</b> before accounting for the additional bundle revenue "
    "($482K modeled above). "
    "The ROI improvement is projected at <b>+15% to +25%</b> vs the current blanket approach "
    "(industry benchmark — requires A/B validation).", s_ok))
story.append(PageBreak())

# ── SECTION 6: RECOMMENDATIONS & ROADMAP ──────────────────────────────────────
story.append(Paragraph("6. Strategic Recommendations & Roadmap", s_h1))

recs = [
    ("R1 — Launch Citrus &amp; Organic Bundles Immediately", s_ok,
     "Deploy the 3 priority rules: Limes+Large Lemon ($71K), Avocado+Bananas ($70K), "
     "Raspberries+Strawberries ($42K). Modeled total gain: $183K. "
     "Format: 'Frequently Bought Together' widget on app/site + adjacent shelf placement. "
     "Apply a 5% discount for Segment 2 only. "
     "Expected basket size increase: +8% to +12%."),
    ("R2 — Build a Premium Loyalty Program for Segment 2", s_ok,
     "Heavy Buyers ($130 avg. spend, 30 items/basket) are your most profitable customers. "
     "A VIP program — organic product cashback, early promo access, priority service — "
     "costs little but dramatically improves customer lifetime value. "
     "Losing even 10% of Segment 2 has more revenue impact than losing all of "
     "Segments 1 and 3 combined."),
    ("R3 — Replace ALL Blanket Discounts with Segment-Specific Offers", s_warn,
     "Eliminate the 15%-for-everyone approach. Replace with: premium organic bundles for Seg. 2, "
     "5% discovery offers for Seg. 1 &amp; 3, upsell recommendations for Seg. 0. "
     "Projected promotional ROI improvement: +15% to +25% per campaign."),
    ("R4 — Automate the Anti-Churn Sequence (Day +15 Trigger)", s_warn,
     "The median inter-purchase interval is 15 days. Set up automated triggers: "
     "Day +15: basket reminder push; Day +21: personalized offer based on RFM history (-8%); "
     "Day +30: strong recovery coupon (-15%, 72h urgency). "
     "Goal: recover 10-15% of silent customers before permanent churn."),
    ("R5 — Rationalize Catalog to the Pareto Core", s_warn,
     "Only 4,778 products (14.6% of the catalog) generate 80% of sales. "
     "Products with low sales volume AND low reorder rates should be evaluated "
     "for delisting. This reduces inventory costs, simplifies logistics, and improves "
     "shelf visibility for star products."),
]

for title, style, body in recs:
    story.append(Paragraph(f"<b>{title}</b>", s_h2))
    story.append(Paragraph(body, style))
    story.append(Spacer(1, 0.2*cm))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("Implementation Roadmap", s_h2))
roadmap_table = [
    ['Phase', 'Timeline', 'Actions', 'Expected Outcome'],
    ['1 — Quick Wins',    'Month 1',   'Top 5 bundles deployed; shelf repositioning; push campaign at peak hour/Monday',
     'Basket +8%, incremental bundle revenue'],
    ['2 — Targeting',     'Month 2',   'End of blanket discounts; 4-segment email campaigns; anti-churn automation',
     'Promo ROI +15%; churn -10%'],
    ['3 — Automation',    'Month 3',   'Monthly pipeline refresh; live KPI dashboard; A/B test ±2h around peak',
     'Continuous insight generation'],
    ['4 — Scale',         'Month 4-6', 'Bundle expansion; Seg. 2 VIP loyalty; catalog rationalization',
     'Revenue target: +12% to +20%'],
]
story.append(make_table(roadmap_table, [2.5*cm, 2*cm, 7*cm, 5*cm], '#2E86AB'))
story.append(Spacer(1, 0.8*cm))

story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
story.append(Paragraph(
    "<i>Limitations &amp; Assumptions — Prices are estimated (uniform random by department; "
    "no real prices in the Instacart dataset). Lift assumes stable purchasing behavior; "
    "seasonal trends may affect rule reliability. The 25% basket lift assumption for targeted "
    "promotions is an industry benchmark and must be validated by A/B testing before full deployment. "
    "Customer segments should be refreshed every 60-90 days. "
    "All financial figures in this report are modeled estimates for strategic purposes only.</i>",
    s_note))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "DSTI Applied MSc — Retail Insights Report · Dataset: Instacart Market Basket Analysis · "
    "Pipeline: FP-Growth + KMeans + RFM · Generated 2025", s_note))

doc.build(story)
print("✅ PDF report generated: retail_insights_report_DSTI.pdf")

from google.colab import files
files.download('retail_insights_report_DSTI.pdf')
