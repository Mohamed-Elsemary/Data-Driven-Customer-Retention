import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, html, body { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background: #f7f8fa;
}
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e6ea;
}
section[data-testid="stSidebar"] * { color: #1a1a2e !important; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #2d3748 !important; font-size: 0.92rem; }
[data-testid="stMarkdownContainer"] h1 { color: #1a1a2e !important; font-size: 1.9rem; font-weight: 700; }
[data-testid="stMarkdownContainer"] h2 { color: #1a1a2e !important; font-size: 1.3rem; font-weight: 600; }
[data-testid="stMarkdownContainer"] h3 { color: #2d3748 !important; font-size: 1.05rem; font-weight: 600; }
h1, h2, h3 { color: #1a1a2e !important; }
p { color: #2d3748 !important; }

button[data-baseweb="tab"] p { color: #64748b !important; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #2563eb !important; }

div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.8rem; }
div[data-testid="stMetric"] div   { color: #1a1a2e !important; font-weight: 600; }

.kpi {
    background: #ffffff;
    border: 1px solid #e2e6ea;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.kpi-val { font-size: 2rem; font-weight: 700; color: #2563eb; }
.kpi-lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; }
.kpi-sub { font-size: 0.82rem; margin-top: 4px; }
.red  { color: #dc2626; }
.green { color: #16a34a; }

.insight {
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-top: 10px;
    color: #78350f !important;
    font-size: 0.88rem;
}
.insight p { color: #78350f !important; }

.section { font-size: 1.15rem; font-weight: 600; color: #1a1a2e; margin: 24px 0 12px; border-bottom: 2px solid #e2e6ea; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# theme for all plotly charts 
T = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
         plot_bgcolor="rgba(0,0,0,0)", font_color="#2d3748",
         margin=dict(t=40, b=10, l=10, r=10))
RED, BLUE = "#dc2626", "#2563eb"
PALETTE = [BLUE, RED, "#16a34a", "#f59e0b", "#7c3aed"]


# data loading
HERE = os.path.dirname(os.path.abspath(__file__))

def load_df():
    path = os.path.join(HERE, "churn_data_processed.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    

def load_results():
    path = os.path.join(HERE, "model_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    

df = load_df()
results = load_results()



_contract_str_map = {
    "Month-to-month": "Month-to-Month",
    "One year":        "One Year",
    "Two year":        "Two Year",
}
df["Contract_label"] = df["Contract"].map(_contract_str_map).fillna(df["Contract"].astype(str))


df["elec_check"] = df["PaymentMethod_Electronic check"].map({0: "Other", 1: "Elec. Check"})

def inet_type(row):
    if row["InternetService_Fiber optic"] == 1: return "Fiber Optic"
    if row["InternetService_No"] == 1:          return "No Internet"
    return "DSL"

df["inet_type"] = df.apply(inet_type, axis=1)

df["sec_label"] = df["OnlineSecurity"].map({0: "No Security", 1: "Has Security"})

df["tenure_band"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72],
                            labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])

# derived stats
n_total   = len(df)
n_churned = int(df["Churn"].sum())
churn_pct = n_churned / n_total * 100
avg_bill  = df["MonthlyCharges"].mean()
total_rev = df["TotalCharges"].sum()
best      = results.loc[results["ROC-AUC"].idxmax()]
using_csv = os.path.exists(os.path.join(HERE, "churn_data_processed.csv"))



# sidebar 
with st.sidebar:
    st.markdown("### Churn Dashboard")
    st.markdown("---")
    page = st.radio("Navigation", ["Overview", "EDA", "Model Results", "Business Insights"],
                    label_visibility="collapsed")
    st.markdown("---")
    


# helpers
def insight(text):
    st.markdown(f'<div class="insight"><p>{text}</p></div>', unsafe_allow_html=True)

def section(text):
    st.markdown(f'<div class="section">{text}</div>', unsafe_allow_html=True)

def bar(x, y, title, xlabel="", ylabel="", text=None):
    fig = px.bar(x=x, y=y, title=title, labels={"x": xlabel, "y": ylabel}, text=text)
    fig.update_layout(**T)
    fig.update_traces(textposition="outside")
    return fig



# PAGES
#  OVERVIEW
if page == "Overview":
    st.markdown("# Customer Churn Dashboard")
    st.markdown("A summary of churn patterns, key risk factors, and model performance for the retention team.")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, sub, cls in [
        (c1, f"${total_rev/1_000_000:.2f}M", "Total Revenue",     "",                  ""),
        (c2, f"{n_total:,}",           "Total Customers",   "",                  ""),
        (c3, f"{n_churned:,}",         "Churned",           "",                  ""),
        (c4, f"{churn_pct:.1f}%",      "Churn Rate",        "",                  ""),
        (c5, f"${avg_bill:.0f}",       "Avg Monthly Bill",  "",                  ""),
    ]:
        col.markdown(f"""
        <div class="kpi">
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-sub {cls}">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    section("Customer Churn Overview")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        fig = go.Figure(go.Pie(
            values=[n_total - n_churned, n_churned],
            labels=["Retained", "Churned"],
            marker_colors=["#16a34a", RED],
            hole=0.5, textfont_size=13,
        ))
        fig.update_layout(**T, showlegend=True, title="Overall churn split")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        grp = df.groupby("tenure_band", observed=True)["Churn"].mean() * 100
        fig = bar(grp.index.astype(str), grp.values,
                  "Churn rate by customer lifecycle stage",
                  ylabel="Churn rate (%)",
                  text=[f"{v:.1f}%" for v in grp.values])
        fig.update_traces(marker_color=[RED if v > 30 else BLUE for v in grp.values])
        st.plotly_chart(fig, use_container_width=True)

    insight("New customers (under 1 year) churn at nearly 47%. Once a customer passes two years, churn drops below 10%. The first year is where retention effort matters most.")


# EDA
elif page == "EDA":
    st.markdown("# Exploratory Data Analysis")
    st.markdown("Charts that show the main drivers of churn, drawn from the analysis in the notebook.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tenure & Charges", "Contract & Payment", "Internet & Security",
        "Feature Engineering", "Correlations"
    ])

    # Tab 1: Tenure & Charges
    with tab1:
        section("Tenure vs Churn")
        col_a, col_b = st.columns(2)

        with col_a:
            grp = df.groupby("tenure_band", observed=True)["Churn"].agg(["mean", "count"]).reset_index()
            grp.columns = ["Band", "Rate", "Count"]
            grp["Rate"] *= 100
            fig = px.bar(grp, x="Band", y="Rate",
                         text=grp["Rate"].map("{:.1f}%".format),
                         title="Churn rate per tenure band",
                         labels={"Band": "Tenure", "Rate": "Churn rate (%)"},
                         color="Rate", color_continuous_scale="RdYlGn_r")
            fig.update_layout(**T, coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = px.histogram(df, x="MonthlyCharges",
                               color=df["Churn"].map({0: "Retained", 1: "Churned"}),
                               barmode="overlay", nbins=40, opacity=0.7,
                               color_discrete_map={"Retained": BLUE, "Churned": RED},
                               title="Monthly charges: churned vs retained",
                               labels={"MonthlyCharges": "Monthly charges ($)", "color": ""})
            fig.update_layout(**T)
            st.plotly_chart(fig, use_container_width=True)

        insight("New customers churn at ~47% — nearly three times the rate of customers past 2 years. Churners also tend to have higher monthly bills: the median monthly charge for a churner is around $80 vs ~$65 for customers who stay.")

        section("Total Charges vs Churn")
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.box(df, x=df["Churn"].map({0: "Retained", 1: "Churned"}), y="tenure",
                         color=df["Churn"].map({0: "Retained", 1: "Churned"}),
                         color_discrete_map={"Retained": BLUE, "Churned": RED},
                         title="Tenure distribution: churned vs retained",
                         labels={"x": "", "y": "Tenure (months)"})
            fig.update_layout(**T, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.box(df, x=df["Churn"].map({0: "Retained", 1: "Churned"}), y="TotalCharges",
                         color=df["Churn"].map({0: "Retained", 1: "Churned"}),
                         color_discrete_map={"Retained": BLUE, "Churned": RED},
                         title="Total charges distribution",
                         labels={"x": "", "y": "Total charges ($)"})
            fig.update_layout(**T, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        insight("Total charges for churners cluster near zero — not because they pay less monthly, but because they leave so early that they never accumulate a high total. This confirms that the real risk window is the first few months.")

    # Tab 2: Contract & Payment
    with tab2:
        section("Contract Type")
        col_a, col_b = st.columns(2)
        with col_a:
            cr = df.groupby("Contract_label")["Churn"].mean().dropna() * 100
            cr = cr.reset_index(); cr.columns = ["Contract", "Rate"]
            cr = cr.sort_values("Rate", ascending=False)
            fig = px.bar(cr, x="Contract", y="Rate",
                         text=cr["Rate"].map("{:.1f}%".format),
                         title="Churn rate by contract type",
                         labels={"Contract": "", "Rate": "Churn rate (%)"},
                         color="Rate", color_continuous_scale="RdYlGn_r")
            fig.update_layout(**T, coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            pivot = df.groupby(["Contract_label", "inet_type"])["Churn"].mean().unstack().fillna(0) * 100
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale="RdYlGn_r", zmin=0, zmax=70,
                text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}", showscale=True,
            ))
            fig.update_layout(**T, title="Churn rate — contract x internet service",
                              xaxis_title="Internet service", yaxis_title="Contract type")
            st.plotly_chart(fig, use_container_width=True)

        insight("Month-to-month customers churn at over 40%. On a two-year contract, that number collapses to near zero. The worst combination: Fiber optic internet on a month-to-month contract sits at 54.6% churn — these customers are expensive to serve and have zero switching cost.")

        section("Payment Method")
        col_a, col_b = st.columns(2)
        with col_a:
            pm = df.groupby("elec_check")["Churn"].mean() * 100
            colors = [RED if "Check" in i else BLUE for i in pm.index]
            fig = go.Figure(go.Bar(x=pm.index.tolist(), y=pm.values,
                                   marker_color=colors,
                                   text=[f"{v:.1f}%" for v in pm.values],
                                   textposition="outside"))
            fig.update_layout(**T, title="Electronic check vs other payment methods",
                              yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            pivot2 = df.groupby(["Contract_label", "elec_check"])["Churn"].mean().unstack().fillna(0) * 100
            fig = go.Figure(go.Heatmap(
                z=pivot2.values, x=pivot2.columns.tolist(), y=pivot2.index.tolist(),
                colorscale="RdYlGn_r", zmin=0, zmax=70,
                text=[[f"{v:.1f}%" for v in row] for row in pivot2.values],
                texttemplate="%{text}", showscale=True,
            ))
            fig.update_layout(**T, title="Churn rate — contract x payment method",
                              xaxis_title="Payment method", yaxis_title="Contract type")
            st.plotly_chart(fig, use_container_width=True)

        insight("Electronic check users churn at 45.3% — nearly three times higher than customers on automatic payment methods. The likely cause: manual payers see their bill every month and that reminder triggers cancellation. AutoPay removes that friction.")

    # Tab 3: Internet & Security
    with tab3:
        section("Internet Service Type")
        col_a, col_b = st.columns(2)
        with col_a:
            ci = df.groupby("inet_type")["Churn"].mean() * 100
            fig = px.bar(x=ci.index.tolist(), y=ci.values,
                         text=[f"{v:.1f}%" for v in ci.values],
                         title="Churn rate by internet service type",
                         labels={"x": "Internet service", "y": "Churn rate (%)"},
                         color=ci.values, color_continuous_scale="RdYlGn_r")
            fig.update_layout(**T, coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.box(df, x="inet_type", y="MonthlyCharges",
                         color="inet_type", color_discrete_sequence=PALETTE,
                         title="Monthly charges by internet service",
                         labels={"inet_type": "Internet service", "MonthlyCharges": "Monthly charges ($)"})
            fig.update_layout(**T, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        insight("Fiber optic customers churn at the highest rate even though — or because — they pay the most. Customers without internet (phone-only) are the most stable. Fiber optic on a month-to-month contract is the single riskiest profile in the dataset.")

        section("Online Security Add-on")
        col_a, col_b = st.columns(2)
        with col_a:
            cs = df.groupby("sec_label")["Churn"].mean() * 100
            colors = [RED if "No" in i else BLUE for i in cs.index]
            fig = go.Figure(go.Bar(x=cs.index.tolist(), y=cs.values,
                                   marker_color=colors,
                                   text=[f"{v:.1f}%" for v in cs.values],
                                   textposition="outside"))
            fig.update_layout(**T, title="Churn rate: online security subscription",
                              yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            # senior citizen churn
            sc = df.groupby("SeniorCitizen")["Churn"].mean() * 100
            fig = go.Figure(go.Bar(
                x=["Not Senior", "Senior"],
                y=sc.values,
                marker_color=[BLUE, RED],
                text=[f"{v:.1f}%" for v in sc.values],
                textposition="outside"
            ))
            fig.update_layout(**T, title="Churn rate: senior vs non-senior citizens",
                              yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        insight("Customers without online security churn at roughly double the rate of those who have it. Bundling security into plans for at-risk customers could be an effective low-cost retention lever. Senior citizens also churn noticeably more than younger customers.")

    # Tab 4: Feature Engineering
    with tab4:
        section("Add-on Services")
        if "_Total_AddOns-Services" in df.columns:
            addon_churn = df.groupby("_Total_AddOns-Services")["Churn"].mean() * 100
            fig = px.line(x=addon_churn.index, y=addon_churn.values, markers=True,
                          title="Churn rate by number of add-on services",
                          labels={"x": "Number of add-on services", "y": "Churn rate (%)"})
            fig.update_traces(line_color=RED, marker_color=BLUE)
            fig.update_layout(**T)
            st.plotly_chart(fig, use_container_width=True)
            insight("Customers with few add-on services who are also paying high monthly rates feel they are not getting value — and they churn. As add-on count increases, churn and no-churn groups converge: high-spend customers with many services feel the cost is justified.")

        section("AutoPay vs Manual Payment")
        if "_Is_AutoPay" in df.columns:
            ap = df.groupby("_Is_AutoPay")["Churn"].mean() * 100
            fig = go.Figure(go.Bar(
                x=["Manual Payment", "AutoPay"],
                y=ap.values,
                marker_color=[RED, BLUE],
                text=[f"{v:.1f}%" for v in ap.values],
                textposition="outside"
            ))
            fig.update_layout(**T, title="Churn rate: AutoPay vs manual payment",
                              yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)
            insight("Manual payment methods roughly double churn risk compared to automatic ones. AutoPay keeps customers passively engaged — they never have to think about cancelling.")

        section("Loyalty Score")
        if "_LoyaltyScore" in df.columns:
            loy = df.groupby("_LoyaltyScore")["Churn"].mean() * 100
            fig = go.Figure(go.Bar(
                x=["Not Loyal", "Loyal"],
                y=loy.values,
                marker_color=[RED, BLUE],
                text=[f"{v:.1f}%" for v in loy.values],
                textposition="outside"
            ))
            fig.update_layout(**T, title="Churn rate: loyal vs non-loyal customers",
                              yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)
            insight("Non-loyal customers churn at ~31%. Loyal customers (long tenure + low charges) churn at just 4.9%. Loyalty score is one of the strongest single-feature predictors in the model.")

        section("Household Stability")
        if "_Household_Stability" in df.columns:
            hs = df.groupby("_Household_Stability")["Churn"].mean() * 100
            fig = go.Figure(go.Bar(
                x=["Alone", "Partner or Dependents", "Full Family"],
                y=hs.values,
                marker_color=[RED, "#f59e0b", BLUE],
                text=[f"{v:.1f}%" for v in hs.values],
                textposition="outside"
            ))
            fig.update_layout(**T, title="Churn rate by household size",
                              yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)
            insight("Household size is inversely correlated with churn. Customers who live alone are the most likely to leave. Customers with both a partner and dependents are the stickiest. Family obligations reduce the likelihood of switching providers.")

        section("High Friction Payment")
        if "_HighFriction_Payment" in df.columns:
            hf = df.groupby("_HighFriction_Payment")["Churn"].mean() * 100
            fig = go.Figure(go.Bar(
                x=["Low Friction", "High Friction"],
                y=hf.values,
                marker_color=[BLUE, RED],
                text=[f"{v:.1f}%" for v in hf.values],
                textposition="outside"
            ))
            fig.update_layout(**T, title="Churn rate: high friction vs low friction payment",
                              yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)
            insight("Customers who use both paperless billing and electronic check (high friction) churn significantly more. They receive digital bills and must manually pay every month — a cycle that keeps churn intent top of mind.")

    # Tab 5: Correlations
    with tab5:
        section("Feature Correlations with Churn")
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn",
                    "SeniorCitizen", "PaperlessBilling", "OnlineSecurity"]
        available = [c for c in num_cols if c in df.columns]
        corr = df[available].corr(method="spearman")

        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", showscale=True,
        ))
        fig.update_layout(**T, title="Spearman correlation matrix")
        st.plotly_chart(fig, use_container_width=True)

        churn_corr = corr["Churn"].drop("Churn").sort_values()
        colors = [RED if v > 0 else BLUE for v in churn_corr.values]
        fig2 = go.Figure(go.Bar(
            x=churn_corr.values, y=churn_corr.index,
            orientation="h", marker_color=colors,
            text=[f"{v:.3f}" for v in churn_corr.values],
            textposition="outside",
        ))
        fig2.update_layout(**{k: v for k, v in T.items() if k != "margin"},
                           title="Each feature's correlation with Churn",
                           xaxis_title="Spearman correlation",
                           margin=dict(t=40, b=10, l=160, r=60))
        st.plotly_chart(fig2, use_container_width=True)
        insight("Contract type and tenure are the strongest negative correlates with churn — the longer someone is committed, the less likely they are to leave. Monthly charges and paperless billing show positive correlation with churn.")


# MODEL RESULTS
elif page == "Model Results":
    st.markdown("# Model Performance")
    st.markdown("Comparison of all five classifiers on test data.")

    metrics = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"] if c in results.columns]

    section("Radar Comparison")
    fig = go.Figure()
    for i, (_, row) in enumerate(results.iterrows()):
        vals = [row[m] for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=metrics + [metrics[0]],
            fill="toself", name=row["Model"],
            line_color=PALETTE[i % len(PALETTE)], opacity=0.7
        ))
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", font_color="#2d3748",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", y=-0.2), margin=dict(t=30, b=60),
        title="All models across all metrics"
    )
    st.plotly_chart(fig, use_container_width=True)

    section("Metric Breakdown")
    fig2 = go.Figure()
    for m, color in zip(metrics, PALETTE):
        fig2.add_trace(go.Bar(name=m, x=results["Model"], y=results[m],
                              marker_color=color,
                              text=[f"{v:.3f}" for v in results[m]],
                              textposition="outside"))
    fig2.update_layout(**{k: v for k, v in T.items() if k != "margin"},
                       barmode="group", yaxis_range=[0, 1.1],
                       legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=80))
    st.plotly_chart(fig2, use_container_width=True)

    section("Scorecard")
    sort_by = st.selectbox("Sort by", metrics,
                           index=metrics.index("ROC-AUC") if "ROC-AUC" in metrics else 0)
    disp = results.sort_values(sort_by, ascending=False).reset_index(drop=True)

    def highlight(s):
        return ["background-color:#dcfce7; color:#166534; font-weight:600"
                if v == s.max() else "" for v in s]

    st.dataframe(
        disp.style.apply(highlight, subset=metrics).format({m: "{:.4f}" for m in metrics}),
        use_container_width=True, height=210
    )

    b = results.loc[results["ROC-AUC"].idxmax()]
    st.success(f"Best model: {b['Model']} — AUC {b['ROC-AUC']:.4f} | F1 {b['F1']:.4f} | Recall {b['Recall']:.4f}")

    section("Why these metrics matter")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Recall** — How many actual churners does the model catch?  
For a telecom, a missed churner means lost revenue with no chance to intervene.

**F1 Score** — A balance between catching real churners and avoiding false alarms.
Too many alerts waste money; too few miss customers who will leave.
        """)
    with col_b:
        st.markdown("""
**ROC-AUC** — How well does the model separate churners from loyal customers?  
This is the primary ranking metric.

**Precision** — Of all predicted churners, how many actually churned?  
Matters when retention offers (discounts, calls) carry a real cost per contact.
        """)


# BUSINESS INSIGHTS
elif page == "Business Insights":
    st.markdown("# Business Insights")
    st.markdown("What the data and model mean for the business, and what actions to take.")

    section("Customer Risk Segmentation")

    def risk_score(row):
        s = 0
        try:
            if int(float(row.get("Contract", 1))) == 0: s += 3
        except Exception:
            if "Month" in str(row.get("Contract_label", "")): s += 3
        try:
            if int(float(row.get("PaymentMethod_Electronic check", 0))) == 1: s += 2
        except Exception:
            pass
        if row.get("tenure", 24) <= 12: s += 3
        try:
            if int(float(row.get("OnlineSecurity", 1))) == 0: s += 1
        except Exception:
            pass
        try:
            if int(float(row.get("InternetService_Fiber optic", 0))) == 1: s += 2
        except Exception:
            pass
        if s >= 7: return "High Risk"
        if s >= 4: return "Medium Risk"
        return "Low Risk"

    df["risk"] = df.apply(risk_score, axis=1)
    risk_counts = df["risk"].value_counts().reset_index()
    risk_counts.columns = ["Segment", "Count"]
    risk_churn = df.groupby("risk")["Churn"].mean() * 100
    risk_counts["Churn Rate"] = risk_counts["Segment"].map(risk_churn)

    col_a, col_b = st.columns(2)
    with col_a:
        color_map = {"High Risk": RED, "Medium Risk": "#f59e0b", "Low Risk": BLUE}
        fig = px.pie(risk_counts, values="Count", names="Segment",
                     color="Segment", color_discrete_map=color_map,
                     title="Customer distribution by risk segment")
        fig.update_layout(**T)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.bar(risk_counts, x="Segment", y="Churn Rate", color="Segment",
                     color_discrete_map=color_map,
                     title="Churn rate per segment",
                     text=risk_counts["Churn Rate"].map("{:.1f}%".format),
                     labels={"Churn Rate": "Churn rate (%)"})
        fig.update_layout(**T, showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    section("Revenue at Risk")
    high = df[df["risk"] == "High Risk"]
    med  = df[df["risk"] == "Medium Risk"]
    r_high, r_med = high["MonthlyCharges"].sum(), med["MonthlyCharges"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk monthly revenue", f"${r_high:,.0f}", f"{len(high):,} customers")
    c2.metric("Medium-risk monthly revenue", f"${r_med:,.0f}", f"{len(med):,} customers")
    c3.metric("Total at-risk revenue", f"${r_high + r_med:,.0f}", "if no action taken")

    section("What the data tells us — and what to do about it")

    findings = [
        ("Early tenure is the highest-risk window",
         "Nearly half of customers who leave do so in their first year. The product or onboarding experience is not meeting expectations quickly enough. "
         "Action: introduce a structured 90-day onboarding programme with proactive check-ins at month 1 and month 3."),
        ("Month-to-month contracts with Fiber optic are the worst combination",
         "54.6% of customers on this profile churn — the highest of any segment. Fiber optic is expensive and month-to-month gives them zero switching cost. "
         "Action: offer a first-year discount specifically for Fiber optic customers who commit to a one-year contract at sign-up."),
        ("Electronic check is a churn signal in itself",
         "Customers paying by electronic check churn at 45.3%. Manual billing creates a monthly reminder of cost. "
         "Action: run an AutoPay enrolment campaign with a $5/month credit for switching. This is one of the cheapest retention levers available."),
        ("Customers without security add-ons churn at twice the rate",
         "OnlineSecurity is a strong protective factor. Customers who have it feel they get tangible value from the service. "
         "Action: bundle a free 3-month security trial into plans for new customers or high-risk profiles."),
        ("Household stability predicts loyalty",
         "Customers who live alone churn the most. Families — partners and dependents — are the stickiest. "
         "Action: design family plan incentives and referral programmes targeting solo customers to increase switching cost."),
        ("AutoPay and loyalty score are the strongest protective features",
         "Loyal customers (long tenure, reasonable monthly charge) on AutoPay churn at under 5%. These are customers to reward, not just retain. "
         "Action: introduce a loyalty reward at the 24-month milestone to lock in the behaviour that is already working."),
    ]

    for title, body in findings:
        with st.expander(title):
            st.markdown(body)

#     section("Deployment plan")
#     b = results.loc[results["ROC-AUC"].idxmax()]
#     st.markdown(f"""
# **Recommended model:** {b['Model']} (AUC {b['ROC-AUC']:.3f}, Recall {b.get('Recall', 0.6):.0%})

# Score every customer weekly. Flag anyone with a churn probability above 0.55 for outreach.  
# Rank flagged customers by: **monthly revenue x churn probability** to prioritise who the retention team contacts first.  
# Log every intervention outcome and retrain the model quarterly with the updated data.

# At a 35% successful intervention rate on high-risk customers, the model protects an estimated  
# **${r_high * 0.35:,.0f}/month** in revenue that would otherwise be lost.
#     """)