# app.py (full clean version)
import os
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Antibiogram 2025", page_icon="🧫", layout="wide")

DATA_PATH = os.path.join("data", "Antibiogram_2025_unified_long.csv")

SOURCE_TO_SPECIMEN = {
    "Pneumonia": "Sputum",
    "UTI": "Urine",
    "Unknown Septicemia": "Blood",
}

# Reserve / last-line toggle keywords (edit freely)
RESERVE_ABX_KEYWORDS = ["COLISTIN", "POLYMYXIN", "TIGECYCLINE", "LINEZOLID", "DAPTOMYCIN"]

def is_reserve_abx(abx: str) -> bool:
    if not isinstance(abx, str):
        return False
    u = abx.upper()
    return any(k in u for k in RESERVE_ABX_KEYWORDS)

# A. baumannii / Acb complex detection
A_BAUM_PATTERNS = [
    r"\bacinetobacter\s+baumannii\b",
    r"\ba\.\s*baumannii\b",
    r"\bcalcoaceticus[-\s]*baumannii\b",
    r"\bacinetobacter\s+calcoaceticus[-\s]*baumannii\s+complex\b",
]

def is_abaum(name: str) -> bool:
    if not isinstance(name, str):
        return False
    s = name.strip().lower()
    return any(re.search(p, s, flags=re.IGNORECASE) for p in A_BAUM_PATTERNS)

def categorize_susceptibility(x: float) -> str:
    if pd.isna(x):
        return "No data"
    if x > 70:
        return "Recommended"
    if 50 <= x <= 70:
        return "Borderline"
    return "Not Recommended"

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Susceptibility_%"] = pd.to_numeric(df["Susceptibility_%"], errors="coerce")
    df["Total_isolates"] = pd.to_numeric(df["Total_isolates"], errors="coerce")

    for c in ["Specimen", "Setting_cat", "Organism_group", "Organism", "Antibiotic"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["Specimen"] = df["Specimen"].replace({"All": "All Specimen"})
    df["__is_abaum__"] = df["Organism"].apply(is_abaum)
    df["__is_reserve__"] = df["Antibiotic"].apply(is_reserve_abx)
    return df

def specimen_overall_view(df: pd.DataFrame, specimen: str) -> pd.DataFrame:
    return df[(df["Specimen"] == specimen) & (df["Setting_cat"] == "Overall")].copy()

def top_pathogens(df_view: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return (
        df_view.dropna(subset=["Total_isolates"])
        .groupby("Organism", as_index=False)["Total_isolates"]
        .max()
        .sort_values("Total_isolates", ascending=False)
        .head(n)
    )

def traffic_style_from_value(v: float) -> str:
    """Cell background color for % susceptibility."""
    try:
        if pd.isna(v):
            return "color:#999;"
        v = float(v)
    except Exception:
        return ""
    if v > 70:
        return "background-color: rgba(0, 200, 0, 0.18);"
    if 50 <= v <= 70:
        return "background-color: rgba(255, 200, 0, 0.18);"
    return "background-color: rgba(255, 0, 0, 0.14);"

def header_style():
    """Better table header readability."""
    return [
        {"selector": "th", "props": [("font-size", "14px"), ("font-weight", "700"), ("text-align", "left")]},
        {"selector": "td", "props": [("font-size", "13px")]},
    ]

# -----------------------------
# Data prep
# -----------------------------
df_all = load_data()
df_no_abaum = df_all[~df_all["__is_abaum__"]].copy()

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("🧫 Antibiogram 2025")
page = st.sidebar.radio("Navigate", ["Empirical Selector", "Organism Viewer", "Data Explorer", "A. baumannii"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Empirical/Viewer/Explorer exclude A. baumannii/Acb complex by design.")

# =============================
# Empirical Selector (Overall only)
# =============================
if page == "Empirical Selector":
    st.title("Empirical Antibiotics Selector")

    c1, c2 = st.columns([2, 1])
    with c1:
        source = st.selectbox("Source", list(SOURCE_TO_SPECIMEN.keys()), index=0)
    with c2:
        top_n = st.number_input("Top pathogens", min_value=1, max_value=10, value=3, step=1)

    hide_reserve = st.checkbox("Hide reserve / last-line agents", value=True)

    specimen = SOURCE_TO_SPECIMEN[source]
    view = specimen_overall_view(df_no_abaum, specimen)

    if hide_reserve:
        view = view[~view["__is_reserve__"]].copy()

    st.caption(f"Overall • {source} (Specimen = {specimen}) • A. baumannii excluded")

    if view.empty:
        st.error("No data found for this selection.")
        st.stop()

    # Top organisms
    top_orgs = top_pathogens(view, n=int(top_n))
    st.subheader("Most Common Pathogens")
    st.dataframe(top_orgs, use_container_width=True, hide_index=True)

    orgs = top_orgs["Organism"].tolist()
    weights = dict(zip(top_orgs["Organism"], top_orgs["Total_isolates"]))
    total_weight = float(sum(weights.values())) if weights else 0.0

    view_top = view[view["Organism"].isin(orgs)].copy()

    def agg_for_abx(g: pd.DataFrame) -> pd.Series:
        has = g["Susceptibility_%"].notna()

        if has.sum() == 0:
            wmean = np.nan
            isolates_covered = 0.0
        else:
            vals = g.loc[has, "Susceptibility_%"].astype(float)
            w = g.loc[has, "Organism"].map(weights).astype(float)
            wmean = float(np.average(vals, weights=w))
            isolates_covered = float(sum(weights.get(o, 0.0) for o in g.loc[has, "Organism"].unique()))

        coverage = (isolates_covered / total_weight * 100.0) if total_weight > 0 else np.nan

        return pd.Series({
            "Weighted Mean (%)": wmean,
            "Isolates Covered": isolates_covered,
            "Coverage (%)": coverage,
        })

    reco = (
        view_top.groupby("Antibiotic", as_index=False)
        .apply(agg_for_abx)
        .reset_index(drop=True)
    )
    reco["Category"] = reco["Weighted Mean (%)"].apply(categorize_susceptibility)

    # Sort by category then score
    order_map = {"Recommended": 0, "Borderline": 1, "Not Recommended": 2, "No data": 3}
    reco["__order__"] = reco["Category"].map(order_map).fillna(99)
    reco = reco.sort_values(["__order__", "Weighted Mean (%)"], ascending=[True, False]).drop(columns="__order__")

    st.subheader("Antibiotic Recommendations")

    styled = (
        reco.style
        .format({"Weighted Mean (%)": "{:.1f}", "Coverage (%)": "{:.1f}", "Isolates Covered": "{:.0f}"})
        .applymap(traffic_style_from_value, subset=["Weighted Mean (%)"])
        .set_table_styles(header_style())
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption("Recommended >70% • Borderline 50–70% • Not Recommended <50% (based on Weighted Mean %)")

    st.download_button(
        "Download recommendation table (CSV)",
        data=reco.to_csv(index=False).encode("utf-8"),
        file_name=f"empirical_{source.lower().replace(' ','_')}_overall.csv",
        mime="text/csv",
    )

# =============================
# Organism Viewer (organism-centric)
# =============================
elif page == "Organism Viewer":
    st.title("Organism-centric Sensitivity Viewer")
    st.caption("Pick an organism → see susceptibilities across specimens (Overall only). A. baumannii excluded.")

    df_spec = df_no_abaum[(df_no_abaum["Setting_cat"] == "Overall") & (df_no_abaum["Specimen"] != "All Specimen")].copy()
    if df_spec.empty:
        st.error("No specimen-overall data available.")
        st.stop()

    organisms = sorted(df_spec["Organism"].dropna().unique().tolist())
    org = st.selectbox("Organism", organisms, index=0)

    org_df = df_spec[df_spec["Organism"] == org].copy()
    available_specs = sorted(org_df["Specimen"].unique().tolist())
    specs_selected = st.multiselect("Specimens", available_specs, default=available_specs)

    view_mode = st.selectbox("View", ["Traffic-light matrix", "Ranked bar (combined)"], index=0)

    org_df = org_df[org_df["Specimen"].isin(specs_selected)].copy()
    if org_df.empty:
        st.info("No data for this selection.")
        st.stop()

    pivot = org_df.pivot_table(index="Antibiotic", columns="Specimen", values="Susceptibility_%", aggfunc="mean").sort_index()
    st.subheader(f"{org} — Susceptibility (%) across specimens")

    if view_mode == "Traffic-light matrix":
        st.dataframe(pivot.style.applymap(traffic_style_from_value).set_table_styles(header_style()), use_container_width=True)
    else:
        st.dataframe(pivot, use_container_width=True)

    iso_by_spec = org_df.dropna(subset=["Total_isolates"]).groupby("Specimen", as_index=False)["Total_isolates"].max()
    weights = dict(zip(iso_by_spec["Specimen"], iso_by_spec["Total_isolates"]))

    def wmean_by_spec(g: pd.DataFrame) -> float:
        vals = g["Susceptibility_%"].astype(float)
        w = g["Specimen"].map(weights).astype(float)
        m = vals.notna() & w.notna()
        if m.sum() == 0:
            return np.nan
        return float(np.average(vals[m], weights=w[m]))

    combined = (
        org_df.groupby("Antibiotic", as_index=False)
        .apply(lambda g: pd.Series({"Weighted Mean (%)": wmean_by_spec(g)}))
        .reset_index(drop=True)
        .sort_values("Weighted Mean (%)", ascending=False)
    )
    combined["Category"] = combined["Weighted Mean (%)"].apply(categorize_susceptibility)

    st.subheader("Combined view (weighted across selected specimens)")
    st.dataframe(
        combined.style.format({"Weighted Mean (%)": "{:.1f}"}).applymap(traffic_style_from_value, subset=["Weighted Mean (%)"]).set_table_styles(header_style()),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = combined.dropna(subset=["Weighted Mean (%)"]).set_index("Antibiotic")[["Weighted Mean (%)"]]
    if not chart_df.empty:
        st.subheader("Quick plot (combined weighted mean)")
        st.bar_chart(chart_df, use_container_width=True)

    st.download_button(
        "Download organism combined table (CSV)",
        data=combined.to_csv(index=False).encode("utf-8"),
        file_name=re.sub(r"[^a-zA-Z0-9]+", "_", f"{org}_combined").strip("_").lower() + ".csv",
        mime="text/csv",
    )

# =============================
# Data Explorer
# =============================
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Explore unified long table (A. baumannii excluded).")

    out = df_no_abaum.copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        specimen = st.selectbox("Specimen", ["All"] + sorted(out["Specimen"].unique().tolist()))
    with c2:
        setting = st.selectbox("Setting", ["All"] + sorted(out["Setting_cat"].unique().tolist()))
    with c3:
        org_group = st.selectbox("Organism group", ["All"] + sorted([x for x in out["Organism_group"].dropna().unique().tolist() if x.strip()]))
    with c4:
        min_iso = st.number_input("Min isolates", min_value=0, max_value=100000, value=0, step=1)

    q1, q2, q3 = st.columns([1.2, 1.2, 1])
    with q1:
        org_contains = st.text_input("Organism contains", "")
    with q2:
        abx_contains = st.text_input("Antibiotic contains", "")
    with q3:
        only_tested = st.checkbox("Only show rows with susceptibility data", value=False)

    sort_by = st.selectbox("Sort by", ["Total_isolates (desc)", "Susceptibility_% (desc)", "Organism (A→Z)"], index=0)

    if specimen != "All":
        out = out[out["Specimen"] == specimen]
    if setting != "All":
        out = out[out["Setting_cat"] == setting]
    if org_group != "All":
        out = out[out["Organism_group"] == org_group]
    if min_iso > 0:
        out = out[out["Total_isolates"].fillna(0) >= min_iso]
    if only_tested:
        out = out[out["Susceptibility_%"].notna()]

    if org_contains.strip():
        out = out[out["Organism"].str.contains(org_contains.strip(), case=False, na=False)]
    if abx_contains.strip():
        out = out[out["Antibiotic"].str.contains(abx_contains.strip(), case=False, na=False)]

    if sort_by == "Total_isolates (desc)":
        out = out.sort_values(["Total_isolates", "Organism", "Antibiotic"], ascending=[False, True, True])
    elif sort_by == "Susceptibility_% (desc)":
        out = out.sort_values(["Susceptibility_%", "Organism", "Antibiotic"], ascending=[False, True, True])
    else:
        out = out.sort_values(["Organism", "Antibiotic"], ascending=[True, True])

    cols_to_show = [c for c in out.columns if not c.startswith("__")]
    st.dataframe(out[cols_to_show], use_container_width=True)

    st.download_button(
        "Download filtered CSV",
        data=out[cols_to_show].to_csv(index=False).encode("utf-8"),
        file_name="antibiogram_filtered.csv",
        mime="text/csv",
    )

# =============================
# A. baumannii dedicated
# =============================
else:
    st.title("A. baumannii / Acb complex")
    st.caption("Dedicated view: A. baumannii / Acinetobacter calcoaceticus-baumannii complex only.")

    abaum = df_all[df_all["__is_abaum__"]].copy()
    if abaum.empty:
        st.info("No A. baumannii/Acb complex rows found with current matching rules.")
        st.stop()

    mode = st.radio(
        "View mode",
        ["Setting-based (All Specimen: ICU/Inpatient/Outpatient/Overall)", "Specimen-based (Overall only: Blood/Urine/Sputum/...)"],
        index=0,
        horizontal=True,
    )

    if mode.startswith("Setting-based"):
        view = abaum[abaum["Specimen"] == "All Specimen"].copy()
        setting = st.selectbox("Setting", sorted(view["Setting_cat"].unique().tolist()), index=0)
        view = view[view["Setting_cat"] == setting].copy()
        st.caption(f"View: All Specimen • Setting = {setting}")
    else:
        view = abaum[(abaum["Setting_cat"] == "Overall") & (abaum["Specimen"] != "All Specimen")].copy()
        specimen = st.selectbox("Specimen (Overall only)", sorted(view["Specimen"].unique().tolist()), index=0)
        view = view[view["Specimen"] == specimen].copy()
        st.caption(f"View: Specimen = {specimen} • Setting = Overall")

    org = st.selectbox("Organism label", sorted(view["Organism"].unique().tolist()), index=0)
    show = view[view["Organism"] == org].dropna(subset=["Susceptibility_%"]).copy()
    show["Category"] = show["Susceptibility_%"].apply(categorize_susceptibility)
    show = show.sort_values("Susceptibility_%", ascending=False)

    cols_to_show = [c for c in show.columns if not c.startswith("__")]
    st.subheader("Susceptibility by antibiotic")
    st.dataframe(
        show[cols_to_show]
        .style.format({"Susceptibility_%": "{:.1f}"}).applymap(traffic_style_from_value, subset=["Susceptibility_%"]).set_table_styles(header_style()),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = show.set_index("Antibiotic")[["Susceptibility_%"]]
    if not chart_df.empty:
        st.subheader("Quick plot")
        st.bar_chart(chart_df, use_container_width=True)

    st.download_button(
        "Download this A. baumannii view (CSV)",
        data=show[cols_to_show].to_csv(index=False).encode("utf-8"),
        file_name="abaumannii_view.csv",
        mime="text/csv",
    )
