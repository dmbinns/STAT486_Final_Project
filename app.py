"""
Utah Fitness Business Thriving Explorer
STAT486 Final Project — Streamlit App
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Utah Fitness Business Explorer",
    page_icon="🏋️",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    clf = pd.read_csv("thriving_model/data/features_classifier.csv", dtype={"zip_code": str})
    full = pd.read_csv("thriving_model/data/utah_fitness_v2.csv", dtype={"zip_code": str})

    # Merge name/rating/review_count into clf from full
    extra = full[["id", "name", "city", "rating", "review_count"]].drop_duplicates("id")
    extra = extra.rename(columns={"name": "biz_name"})
    clf = clf.merge(extra, on="id", how="left")
    clf["name"] = clf.get("biz_name", clf.get("name", ""))
    clf["thriving_label"] = clf["is_thriving"].map({1.0: "Thriving", 0.0: "Not Thriving"})
    clf["city"] = clf["city"].fillna("")
    return clf, full

clf_df, full_df = load_data()

NUMERIC_FEATURES = [
    'median_income', 'median_home_value', 'median_age',
    'pct_bachelors', 'pct_prime_gym_age', 'total_pop',
    'competition_1km', 'competition_3km',
    'market_gap', 'gym_density_per_1k', 'income_per_competitor', 'price',
]
CATEGORICAL_FEATURES = ['category_group']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

@st.cache_resource
def train_model():
    df = pd.read_csv("thriving_model/data/features_classifier.csv", dtype={"zip_code": str})
    X = df[ALL_FEATURES]
    y = df['is_thriving'].astype(int)

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ('pre', preprocessor),
        ('model', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        )),
    ])
    model.fit(X, y)
    return model

gbc_model = train_model()

CATEGORY_LABELS = {
    "general_gym": "General Gym",
    "mind_body": "Mind-Body (Yoga/Pilates/Barre)",
    "martial_arts": "Martial Arts",
    "other": "Other (CrossFit/Dance/etc.)",
    "personal_training": "Personal Training Studio",
    "climbing_outdoor": "Climbing / Outdoor",
}

PALETTE = {"Thriving": "#2ecc71", "Not Thriving": "#e74c3c"}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Utah Fitness Business Explorer")
st.caption("Where should you open your next gym? Powered by Yelp + Census data.")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Map", "Best Locations", "Business Types", "Location Scorer"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("All Utah Fitness Businesses")

    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        cat_options = ["All"] + sorted(clf_df["category_group"].unique().tolist())
        cat_filter = st.selectbox(
            "Filter by business type",
            cat_options,
            format_func=lambda x: "All Types" if x == "All" else CATEGORY_LABELS.get(x, x),
        )
    with col_f2:
        status_filter = st.radio(
            "Show", ["All", "Thriving only", "Not Thriving only"], horizontal=True
        )

    map_df = clf_df.copy()
    if cat_filter != "All":
        map_df = map_df[map_df["category_group"] == cat_filter]
    if status_filter == "Thriving only":
        map_df = map_df[map_df["is_thriving"] == 1.0]
    elif status_filter == "Not Thriving only":
        map_df = map_df[map_df["is_thriving"] == 0.0]

    map_df = map_df.dropna(subset=["latitude", "longitude"])
    map_df["category_label"] = map_df["category_group"].map(CATEGORY_LABELS)
    map_df["biz_name"] = map_df["biz_name"].fillna("Unknown")

    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="thriving_label",
        color_discrete_map=PALETTE,
        hover_name="biz_name",
        hover_data={
            "category_label": True,
            "zip_code": True,
            "rating": True,
            "market_gap": ":.0f",
            "competition_3km": True,
            "latitude": False,
            "longitude": False,
            "thriving_label": False,
        },
        labels={
            "category_label": "Type",
            "zip_code": "ZIP",
            "rating": "Rating",
            "market_gap": "Market Gap",
            "competition_3km": "Competitors (3km)",
        },
        mapbox_style="carto-positron",
        zoom=8,
        center={"lat": 40.6, "lon": -111.7},
        height=580,
        size_max=12,
    )
    fig_map.update_layout(legend_title_text="Status", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Businesses shown", len(map_df))
    thriving_shown = int(map_df["is_thriving"].sum())
    col_m2.metric("Thriving", thriving_shown)
    col_m3.metric(
        "Thriving rate",
        f"{thriving_shown / len(map_df) * 100:.0f}%" if len(map_df) > 0 else "—",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BEST LOCATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Best Zip Codes to Open a Fitness Business")

    col_l1, col_l2 = st.columns([1, 1])
    with col_l1:
        cat_loc = st.selectbox(
            "Filter by business type",
            ["All"] + sorted(clf_df["category_group"].unique().tolist()),
            format_func=lambda x: "All Types" if x == "All" else CATEGORY_LABELS.get(x, x),
            key="loc_cat",
        )
    with col_l2:
        min_n = st.slider("Minimum businesses in zip (for reliability)", 1, 10, 3)

    loc_df = clf_df.copy()
    if cat_loc != "All":
        loc_df = loc_df[loc_df["category_group"] == cat_loc]

    zip_stats = (
        loc_df.groupby("zip_code")
        .agg(
            n=("is_thriving", "count"),
            thriving_rate=("is_thriving", "mean"),
            avg_income=("median_income", "mean"),
            avg_market_gap=("market_gap", "mean"),
            avg_competition_3km=("competition_3km", "mean"),
            pct_prime_age=("pct_prime_gym_age", "mean"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        )
        .reset_index()
    )
    # Add city name from most common city in that zip
    city_map = (
        clf_df.dropna(subset=["city"])
        .groupby("zip_code")["city"]
        .agg(lambda x: x.mode()[0] if len(x) > 0 else "")
        .reset_index()
    )
    zip_stats = zip_stats.merge(city_map, on="zip_code", how="left")
    zip_stats = zip_stats[zip_stats["n"] >= min_n].sort_values("thriving_rate", ascending=False)
    zip_stats["thriving_pct"] = (zip_stats["thriving_rate"] * 100).round(0).astype(int)
    zip_stats["avg_income_fmt"] = zip_stats["avg_income"].apply(lambda x: f"${x:,.0f}")
    zip_stats["avg_market_gap_fmt"] = zip_stats["avg_market_gap"].apply(lambda x: f"{x:,.0f}")

    # Bar chart
    top15 = zip_stats.head(15).copy()
    top15["label"] = top15["zip_code"] + " " + top15["city"].fillna("")
    fig_bar = px.bar(
        top15,
        x="thriving_pct",
        y="label",
        orientation="h",
        color="thriving_pct",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        range_color=[0, 100],
        labels={"thriving_pct": "Thriving Rate (%)", "label": ""},
        title="Top 15 Zip Codes by Thriving Rate",
        height=500,
        text="thriving_pct",
    )
    fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_bar.update_layout(coloraxis_showscale=False, yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_bar, use_container_width=True)

    # Table
    st.markdown("**Full rankings**")
    display_cols = {
        "zip_code": "ZIP",
        "city": "City",
        "n": "# Businesses",
        "thriving_pct": "Thriving %",
        "avg_income_fmt": "Median Income",
        "avg_market_gap_fmt": "Market Gap",
        "avg_competition_3km": "Avg Competitors (3km)",
        "pct_prime_age": "% Prime Gym Age (20-44)",
    }
    table_df = zip_stats[list(display_cols.keys())].rename(columns=display_cols)
    table_df["Avg Competitors (3km)"] = table_df["Avg Competitors (3km)"].round(1)
    table_df["% Prime Gym Age (20-44)"] = table_df["% Prime Gym Age (20-44)"].round(1)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BUSINESS TYPES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Which Type of Fitness Business Thrives?")

    cat_stats = (
        clf_df.groupby("category_group")
        .agg(n=("is_thriving", "count"), thriving_rate=("is_thriving", "mean"))
        .reset_index()
        .sort_values("thriving_rate", ascending=False)
    )
    cat_stats["thriving_pct"] = (cat_stats["thriving_rate"] * 100).round(1)
    cat_stats["label"] = cat_stats["category_group"].map(CATEGORY_LABELS)

    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        fig_cat = px.bar(
            cat_stats,
            x="label",
            y="thriving_pct",
            color="thriving_pct",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            range_color=[0, 100],
            text="thriving_pct",
            labels={"label": "", "thriving_pct": "Thriving Rate (%)"},
            title="Thriving Rate by Business Type",
            height=400,
        )
        fig_cat.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_cat.update_layout(coloraxis_showscale=False, xaxis_tickangle=-20)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_t2:
        st.markdown("**Sample sizes**")
        for _, row in cat_stats.iterrows():
            label = row["label"]
            pct = row["thriving_pct"]
            n = int(row["n"])
            color = "#2ecc71" if pct >= 80 else "#f39c12" if pct >= 60 else "#e74c3c"
            st.markdown(
                f"<div style='padding:6px 0'>"
                f"<b>{label}</b><br>"
                f"<span style='color:{color};font-size:1.2em'>{pct:.0f}%</span> thriving "
                f"<span style='color:gray'>({n} businesses)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Thriving Rate: Type × Location Heatmap")

    # Only zip codes with enough data
    heatmap_df = clf_df.copy()
    zip_counts = heatmap_df.groupby("zip_code")["id"].count()
    valid_zips = zip_counts[zip_counts >= 2].index
    heatmap_df = heatmap_df[heatmap_df["zip_code"].isin(valid_zips)]

    pivot = heatmap_df.pivot_table(
        index="zip_code", columns="category_group", values="is_thriving", aggfunc="mean"
    )
    pivot.columns = [CATEGORY_LABELS.get(c, c) for c in pivot.columns]

    # Add city to index label
    city_map2 = clf_df.dropna(subset=["city"]).groupby("zip_code")["city"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else ""
    )
    pivot.index = [f"{z} {city_map2.get(z, '')}" for z in pivot.index]

    fig_heat = px.imshow(
        pivot * 100,
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        zmin=0,
        zmax=100,
        aspect="auto",
        labels={"color": "Thriving %", "x": "Business Type", "y": "ZIP Code"},
        title="Thriving Rate (%) by ZIP × Business Type (gray = no data)",
        height=max(400, len(pivot) * 22),
    )
    fig_heat.update_layout(coloraxis_colorbar_title="Thriving %")
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOCATION SCORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Location Scorer — Should I Open Here?")
    st.markdown(
        "Pick a zip code and business type. The app auto-fills market conditions from "
        "Census + Yelp data and runs them through the trained Gradient Boosting classifier."
    )

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        zip_options = sorted(clf_df["zip_code"].unique().tolist())
        selected_zip = st.selectbox("ZIP Code", zip_options)
    with col_s2:
        selected_cat = st.selectbox(
            "Business Type",
            sorted(clf_df["category_group"].unique().tolist()),
            format_func=lambda x: CATEGORY_LABELS.get(x, x),
        )

    zip_data = clf_df[clf_df["zip_code"] == selected_zip]

    if len(zip_data) == 0:
        st.warning("No data for this ZIP code.")
    else:
        city_name = zip_data["city"].mode()[0] if len(zip_data) > 0 else ""
        profile = zip_data[NUMERIC_FEATURES].mean()

        st.markdown(f"### {selected_zip} — {city_name}")
        st.markdown(f"**Business type:** {CATEGORY_LABELS.get(selected_cat, selected_cat)}")
        st.markdown("**Adjust market conditions** (pre-filled from zip data):")

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            comp_1km = st.number_input("Competitors within 1km", min_value=0, max_value=50,
                                       value=int(profile["competition_1km"]))
            comp_3km = st.number_input("Competitors within 3km", min_value=0, max_value=100,
                                       value=int(profile["competition_3km"]))
        with col_a2:
            price = st.selectbox("Price tier", [0, 1, 2, 3, 4],
                                 index=int(profile["price"]) if not np.isnan(profile["price"]) else 0,
                                 format_func=lambda x: ["Unknown", "$", "$$", "$$$", "$$$$"][x])
        with col_a3:
            total_pop = st.number_input("ZIP population", min_value=0, max_value=200000,
                                        value=int(profile["total_pop"]))

        # Recompute derived features from adjustments
        gyms_in_zip = comp_3km  # proxy
        market_gap = profile["pct_prime_gym_age"] / 100 * total_pop / (gyms_in_zip + 1)
        gym_density = comp_3km / (total_pop / 1000) if total_pop > 0 else 0
        income_per_comp = profile["median_income"] / (comp_3km + 1)

        input_row = pd.DataFrame([{
            "median_income": profile["median_income"],
            "median_home_value": profile["median_home_value"],
            "median_age": profile["median_age"],
            "pct_bachelors": profile["pct_bachelors"],
            "pct_prime_gym_age": profile["pct_prime_gym_age"],
            "total_pop": total_pop,
            "competition_1km": comp_1km,
            "competition_3km": comp_3km,
            "market_gap": market_gap,
            "gym_density_per_1k": gym_density,
            "income_per_competitor": income_per_comp,
            "price": price,
            "category_group": selected_cat,
        }])

        thriving_prob = gbc_model.predict_proba(input_row)[0][1]

        st.divider()

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(thriving_prob * 100, 1),
            number={"suffix": "%"},
            title={"text": "Predicted Thriving Probability", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if thriving_prob >= 0.65 else "#f39c12" if thriving_prob >= 0.45 else "#e74c3c"},
                "steps": [
                    {"range": [0, 45], "color": "#fde8e8"},
                    {"range": [45, 65], "color": "#fef3cd"},
                    {"range": [65, 100], "color": "#d4f5e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": thriving_prob * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin={"t": 40, "b": 0, "l": 40, "r": 40})

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            if thriving_prob >= 0.65:
                verdict, verdict_color, verdict_icon = "Strong opportunity", "#2ecc71", "✅"
            elif thriving_prob >= 0.45:
                verdict, verdict_color, verdict_icon = "Moderate opportunity", "#f39c12", "⚠️"
            else:
                verdict, verdict_color, verdict_icon = "Challenging market", "#e74c3c", "❌"

            st.markdown(
                f"<div style='background:{verdict_color}22;border-left:5px solid {verdict_color};"
                f"padding:16px;border-radius:4px;margin-top:40px'>"
                f"<h3 style='color:{verdict_color};margin:0'>{verdict_icon} {verdict}</h3>"
                f"<p style='margin:8px 0 0'>The Gradient Boosting classifier gives a "
                f"<b>{thriving_prob * 100:.1f}%</b> chance this business thrives, "
                f"based on the zip code's demographics, competition density, and business type.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Key inputs used:**")
            st.markdown(f"- Market gap: **{market_gap:,.0f}**")
            st.markdown(f"- Income per competitor: **${income_per_comp:,.0f}**")
            st.markdown(f"- Prime gym age %: **{profile['pct_prime_gym_age']:.1f}%**")
            st.markdown(f"- Median income: **${profile['median_income']:,.0f}**")

        st.divider()
        st.markdown("**Businesses already in this ZIP**")
        biz_table = zip_data[["biz_name", "category_group", "thriving_label", "rating", "review_count"]].copy()
        biz_table["category_group"] = biz_table["category_group"].map(CATEGORY_LABELS)
        biz_table.columns = ["Name", "Type", "Status", "Rating", "Reviews"]
        st.dataframe(biz_table.sort_values("Status"), use_container_width=True, hide_index=True)
