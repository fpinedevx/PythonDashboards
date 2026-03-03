import pandas as pd
import numpy as np

from dash import Dash, Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# ---- scikit-learn ----
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
    silhouette_score,
)

# ---- hierarchical clustering ----
from scipy.cluster.hierarchy import linkage
from scipy import stats

# ----------------------------
# Data
# ----------------------------
FILE_PATH = "OfficeSuppliesOrderData.xlsx"  # adjust if needed
df = pd.read_excel(FILE_PATH)

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Year"] = df["Order Date"].dt.year
df["Quarter"] = "Q" + df["Order Date"].dt.quarter.astype("Int64").astype(str)
df["MonthNum"] = df["Order Date"].dt.month.astype("Int64")
df["Month"] = df["Order Date"].dt.strftime("%b")
df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)

# ----------------------------
# Theme URL
# ----------------------------
LIGHT_CSS = "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/spacelab/bootstrap.min.css"

# ----------------------------
# Helpers
# ----------------------------
def apply_filters(data: pd.DataFrame, year, quarter, monthnum, date_preset, regions=None, categories=None, segments=None) -> pd.DataFrame:
    out = data.copy()

    max_dt = out["Order Date"].max()
    if pd.notna(max_dt) and date_preset and date_preset != "ALL_TIME":
        if date_preset == "LAST_60D":
            start = max_dt - pd.Timedelta(days=60)
        elif date_preset == "LAST_90D":
            start = max_dt - pd.Timedelta(days=90)
        elif date_preset == "LAST_180D":
            start = max_dt - pd.Timedelta(days=180)
        elif date_preset == "LAST_1Y":
            start = max_dt - pd.DateOffset(years=1)
        elif date_preset == "LAST_2Y":
            start = max_dt - pd.DateOffset(years=2)
        elif date_preset == "LAST_3Y":
            start = max_dt - pd.DateOffset(years=3)
        elif date_preset == "LAST_5Y":
            start = max_dt - pd.DateOffset(years=5)
        else:
            start = None

        if start is not None:
            out = out[(out["Order Date"] >= start) & (out["Order Date"] <= max_dt)]

    if year not in (None, "All"):
        out = out[out["Year"] == int(year)]
    if quarter not in (None, "All"):
        out = out[out["Quarter"] == quarter]
    if monthnum not in (None, "All"):
        out = out[out["MonthNum"] == int(monthnum)]
    if regions:
        out = out[out["Region"].isin(regions)]
    if categories:
        out = out[out["Category"].isin(categories)]
    if segments:
        out = out[out["Segment"].isin(segments)]
    return out


def safe_currency(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "$0.00"
    return f"${x:,.2f}"


def safe_int(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "0"
    return f"{int(x):,}"


def safe_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "0.00%"
    return f"{x*100:,.2f}%"


def compute_kpis(full_data: pd.DataFrame, filtered: pd.DataFrame):
    total_sales = float(filtered["Sales"].sum()) if not filtered.empty else 0.0
    customer_count = int(filtered["Customer ID"].nunique()) if not filtered.empty else 0
    order_count = int(filtered["Order ID"].nunique()) if not filtered.empty else 0

    anchor = (
        full_data["Order Date"].max()
        if filtered.empty or filtered["Order Date"].isna().all()
        else filtered["Order Date"].max()
    )
    if pd.isna(anchor):
        return dict(
            total_sales=0.0,
            customer_count=0,
            order_count=0,
            ytd_sales=0.0,
            lytd_sales=0.0,
            yoy_pct=np.nan,
        )

    anchor = pd.to_datetime(anchor)

    ytd_start = pd.Timestamp(year=anchor.year, month=1, day=1)
    ytd_sales = float(
        full_data.loc[
            (full_data["Order Date"] >= ytd_start) & (full_data["Order Date"] <= anchor), "Sales"
        ].sum()
    )

    ly_anchor = anchor - pd.DateOffset(years=1)
    ly_start = pd.Timestamp(year=ly_anchor.year, month=1, day=1)
    lytd_sales = float(
        full_data.loc[
            (full_data["Order Date"] >= ly_start) & (full_data["Order Date"] <= ly_anchor), "Sales"
        ].sum()
    )

    yoy_pct = (ytd_sales - lytd_sales) / lytd_sales if lytd_sales != 0 else np.nan

    return dict(
        total_sales=total_sales,
        customer_count=customer_count,
        order_count=order_count,
        ytd_sales=ytd_sales,
        lytd_sales=lytd_sales,
        yoy_pct=yoy_pct,
    )


def kpi_card(title: str, value: str):
    return dbc.Card(
        dbc.CardBody(
            [html.Div(title, className="text-muted"), html.H3(value, className="m-0")],
            className="text-center",
        ),
        className="shadow-sm",
    )


def add_linear_trendline(fig: go.Figure, x_dates: pd.Series, y_values: pd.Series, name="Trend"):
    if len(y_values) < 2:
        return fig
    x_ord = x_dates.map(pd.Timestamp.toordinal).astype(float).to_numpy()
    y = y_values.astype(float).to_numpy()
    mask = np.isfinite(x_ord) & np.isfinite(y)
    if mask.sum() < 2:
        return fig
    m, b = np.polyfit(x_ord[mask], y[mask], 1)
    y_hat = m * x_ord + b
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=y_hat,
            mode="lines",
            name=name,
            line=dict(dash="dash"),
            hovertemplate="Trend: %{y:$,.2f}<extra></extra>",
        )
    )
    return fig


def build_model_pipeline(model_name: str):
    cat_cols = ["Ship Mode", "Segment", "Region", "State", "Category", "Sub-Category", "Product ID"]
    num_cols = ["Postal Code"]

    categorical = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]
    )
    pre = ColumnTransformer(
        transformers=[
            ("cat", categorical, [c for c in cat_cols if c in df.columns]),
            ("num", numeric, [c for c in num_cols if c in df.columns]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=1.0, random_state=0)
    else:
        model = RandomForestRegressor(n_estimators=250, random_state=0, n_jobs=-1, min_samples_leaf=2)

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def build_classification_pipeline(data: pd.DataFrame):
    cat_cols = ["Ship Mode", "Segment", "Region", "State", "Category", "Sub-Category", "Product ID"]
    num_cols = ["Postal Code", "Quantity", "Discount", "Profit"]

    cat_present = [c for c in cat_cols if c in data.columns]
    num_present = [c for c in num_cols if c in data.columns]
    if not cat_present and not num_present:
        return None, []

    transformers = []
    if cat_present:
        categorical = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        transformers.append(("cat", categorical, cat_present))
    if num_present:
        numeric = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]
        )
        transformers.append(("num", numeric, num_present))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline(steps=[("preprocess", pre), ("model", model)]), cat_present + num_present


def compute_classification_artifacts(filtered: pd.DataFrame):
    if filtered.empty or "Sales" not in filtered.columns:
        return {"error": "No data available for classification with current filters."}

    data = filtered.copy()
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Sales"])
    if data.shape[0] < 40:
        return {"error": f"Need at least 40 rows for classification diagnostics (found {data.shape[0]})."}

    threshold = float(data["Sales"].quantile(0.75))
    data["HighValueOrder"] = (data["Sales"] >= threshold).astype(int)

    class_counts = data["HighValueOrder"].value_counts()
    if class_counts.size < 2 or int(class_counts.min()) < 8:
        return {"error": "Not enough class balance after thresholding to compute classification charts."}

    pipe, feature_cols = build_classification_pipeline(data)
    if pipe is None or not feature_cols:
        return {"error": "No usable features available for classification."}

    X = data[feature_cols].copy()
    y = data["HighValueOrder"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y,
    )

    pipe.fit(X_train, y_train)
    y_score = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    return {
        "threshold": threshold,
        "positive_rate": float(y.mean()),
        "rows": int(data.shape[0]),
        "y_test": y_test.to_numpy(dtype=int),
        "y_score": np.asarray(y_score, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=int),
    }


def get_feature_names(pipe: Pipeline):
    try:
        ct: ColumnTransformer = pipe.named_steps["preprocess"]
        feature_names = []
        for name, trans, cols in ct.transformers_:
            if name == "remainder" or trans is None:
                continue
            if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                ohe: OneHotEncoder = trans.named_steps["onehot"]
                feature_names.extend(ohe.get_feature_names_out(cols).tolist())
            else:
                feature_names.extend(list(cols))
        return feature_names
    except Exception:
        return None


def monthly_sales_forecast_series(filtered: pd.DataFrame, horizon_months: int = 6):
    if filtered.empty:
        return None, None

    monthly = filtered.groupby("YearMonth", as_index=False)["Sales"].sum().sort_values("YearMonth")
    monthly["ds"] = pd.to_datetime(monthly["YearMonth"] + "-01")
    monthly = monthly.dropna(subset=["ds"]).reset_index(drop=True)

    hist = monthly[["ds", "Sales"]].rename(columns={"Sales": "y"})
    if monthly.shape[0] < 8:
        return hist, None

    y = monthly["Sales"].astype(float).to_numpy()
    t = np.arange(len(monthly), dtype=float)

    w = 2.0 * np.pi / 12.0
    X = np.column_stack([t, np.sin(w * t), np.cos(w * t)])

    model = Ridge(alpha=5.0, random_state=0)
    model.fit(X, y)

    y_hat_in = model.predict(X)
    resid = y - y_hat_in
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    t_f = np.arange(len(monthly), len(monthly) + horizon_months, dtype=float)
    X_f = np.column_stack([t_f, np.sin(w * t_f), np.cos(w * t_f)])
    y_f = model.predict(X_f)

    start = monthly["ds"].iloc[-1] + pd.offsets.MonthBegin(1)
    future_ds = pd.date_range(start=start, periods=horizon_months, freq="MS")

    fc = pd.DataFrame({"ds": future_ds, "yhat": y_f})
    if sigma > 0:
        fc["yhat_upper"] = fc["yhat"] + 1.96 * sigma
        fc["yhat_lower"] = fc["yhat"] - 1.96 * sigma
    else:
        fc["yhat_upper"] = fc["yhat"]
        fc["yhat_lower"] = fc["yhat"]

    return hist, fc


def build_cluster_df(filtered: pd.DataFrame) -> pd.DataFrame:
    required = {"Region", "Category", "Sales"}
    if filtered.empty or not required.issubset(set(filtered.columns)):
        return pd.DataFrame()

    cols = set(filtered.columns)
    has_profit = "Profit" in cols
    has_discount = "Discount" in cols
    has_qty = "Quantity" in cols
    has_orders = "Order ID" in cols
    has_customers = "Customer ID" in cols

    group_cols = ["Region", "Category"]
    agg = {"Sales": "sum"}
    if has_profit:
        agg["Profit"] = "sum"
    if has_discount:
        agg["Discount"] = "mean"
    if has_qty:
        agg["Quantity"] = "sum"
    if has_orders:
        agg["Order ID"] = pd.Series.nunique
    if has_customers:
        agg["Customer ID"] = pd.Series.nunique

    g = filtered.groupby(group_cols, as_index=False).agg(agg)
    if "Order ID" in g.columns:
        g = g.rename(columns={"Order ID": "OrderCount"})
    if "Customer ID" in g.columns:
        g = g.rename(columns={"Customer ID": "CustomerCount"})

    feat_cols = [c for c in ["Sales", "Profit", "OrderCount", "CustomerCount", "Quantity", "Discount"] if c in g.columns]
    if len(feat_cols) < 2 or g.shape[0] < 4:
        return pd.DataFrame()

    X = g[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    k = int(min(5, max(3, g.shape[0] // 3)))  # 3..5
    k = min(k, g.shape[0] - 1) if g.shape[0] > 1 else 1
    if k < 2:
        return pd.DataFrame()

    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    g["Cluster"] = km.fit_predict(Xs).astype(int)
    return g


def build_kmeans_heatmap(clus_df: pd.DataFrame):
    if clus_df is None or clus_df.empty or "Cluster" not in clus_df.columns:
        return None, None
    feat_cols = [c for c in ["Sales", "Profit", "OrderCount", "CustomerCount", "Quantity", "Discount"] if c in clus_df.columns]
    if len(feat_cols) < 2:
        return None, None
    centers = clus_df.groupby("Cluster")[feat_cols].mean().sort_index()
    z = (centers - centers.mean(axis=0)) / centers.std(axis=0, ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)
    return centers, z


def clusters_vs_inertia(clus_df: pd.DataFrame, k_min: int = 2, k_max: int = 10):
    if clus_df is None or clus_df.empty:
        return None

    feat_cols = [c for c in ["Sales", "Profit", "OrderCount", "CustomerCount", "Quantity", "Discount"] if c in clus_df.columns]
    if len(feat_cols) < 2:
        return None

    X = clus_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    n = Xs.shape[0]
    if n < 4:
        return None

    k_max = min(int(k_max), n - 1)
    k_min = max(2, min(int(k_min), k_max))
    if k_max < 2:
        return None

    ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        km.fit(Xs)
        inertias.append(float(km.inertia_))

    return pd.DataFrame({"k": ks, "inertia": inertias})


def clusters_vs_silhouette(clus_df: pd.DataFrame, k_min: int = 2, k_max: int = 10):
    if clus_df is None or clus_df.empty:
        return None

    feat_cols = [c for c in ["Sales", "Profit", "OrderCount", "CustomerCount", "Quantity", "Discount"] if c in clus_df.columns]
    if len(feat_cols) < 2:
        return None

    X = clus_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    n = Xs.shape[0]
    if n < 4:
        return None

    k_max = min(int(k_max), n - 1)
    k_min = max(2, min(int(k_min), k_max))
    if k_max < 2:
        return None

    ks = []
    scores = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(Xs)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or unique_labels.size >= n:
            continue
        ks.append(k)
        scores.append(float(silhouette_score(Xs, labels)))

    if not ks:
        return None

    return pd.DataFrame({"k": ks, "silhouette": scores})


def build_dendrogram(clus_df: pd.DataFrame, template: str):
    if clus_df is None or clus_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template=template,
            title="Hierarchical Clustering Dendrogram",
            annotations=[dict(text="Not enough data to build a dendrogram.", x=0.5, y=0.5, showarrow=False)],
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig

    feat_cols = [c for c in ["Sales", "Profit", "OrderCount", "CustomerCount", "Quantity", "Discount"] if c in clus_df.columns]
    if len(feat_cols) < 2 or clus_df.shape[0] < 4:
        fig = go.Figure()
        fig.update_layout(
            template=template,
            title="Hierarchical Clustering Dendrogram",
            annotations=[dict(text="Need at least 4 points and 2+ numeric features.", x=0.5, y=0.5, showarrow=False)],
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig

    labels = (clus_df["Region"].astype(str) + " | " + clus_df["Category"].astype(str)).tolist()
    X = clus_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    def _linkagefun(_x):
        return linkage(_x, method="ward")

    try:
        fig = ff.create_dendrogram(Xs, labels=labels, linkagefun=_linkagefun, orientation="left")
        fig.update_layout(
            template=template,
            title="Hierarchical Clustering Dendrogram (Ward linkage)",
            margin=dict(l=20, r=20, t=60, b=20),
            height=max(550, min(1600, 18 * len(labels))),
        )
        fig.update_xaxes(title_text="Distance")
        fig.update_yaxes(title_text="")
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            template=template,
            title="Hierarchical Clustering Dendrogram",
            annotations=[dict(text=f"Could not render dendrogram: {e}", x=0.5, y=0.5, showarrow=False)],
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig


# ----------------------------
# App
# ----------------------------
app = Dash(__name__) #, external_stylesheets=[ICONS_CSS])
app.title = "Office Supplies Dashboard"

app.index_string = f"""
<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <link id="bs-light" rel="stylesheet" href="{LIGHT_CSS}">
    <style>
      .chart-border {{
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 10px;
        padding: 8px;
      }}
      .filters-border {{
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 10px;
        padding: 10px;
      }}
      .custom-tabs .tab {{
        padding: 6px 12px !important;
        line-height: 1.1 !important;
      }}
      .custom-tabs .tab--selected {{
        padding: 6px 12px !important;
      }}
    </style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>
"""

years = sorted(df["Year"].dropna().unique().astype(int).tolist())
year_options = [{"label": "All", "value": "All"}] + [{"label": str(y), "value": str(y)} for y in years]
region_options = [{"label": v, "value": v} for v in sorted(df["Region"].dropna().astype(str).unique().tolist())] if "Region" in df.columns else []
category_options = [{"label": v, "value": v} for v in sorted(df["Category"].dropna().astype(str).unique().tolist())] if "Category" in df.columns else []
segment_options = [{"label": v, "value": v} for v in sorted(df["Segment"].dropna().astype(str).unique().tolist())] if "Segment" in df.columns else []

DD_STYLE = {"fontSize": "0.9rem"}
LABEL_CLASS = "mt-2 mb-1"

preset_options = [
    ("Last 60 days", "LAST_60D"),
    ("Last 90 days", "LAST_90D"),
    ("Last 180 days", "LAST_180D"),
    ("Last 1 year", "LAST_1Y"),
    ("Last 2 years", "LAST_2Y"),
    ("Last 3 years", "LAST_3Y"),
    ("Last 5 years", "LAST_5Y"),
    ("All time", "ALL_TIME"),
]

model_options = ["Linear Regression", "Ridge", "Random Forest"]

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id="ml-store"),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H5("Filters", className="mt-1 mb-2"),
                            dbc.Label("Date preset", className=LABEL_CLASS),
                            dcc.Dropdown(
                                id="date-preset",
                                options=[{"label": l, "value": v} for l, v in preset_options],
                                value="ALL_TIME",
                                clearable=False,
                                style=DD_STYLE,
                            ),
                            dbc.Label("Year", className=LABEL_CLASS),
                            dcc.Dropdown(id="year-dd", options=year_options, value="All", clearable=False, style=DD_STYLE),
                            dbc.Label("Quarter", className=LABEL_CLASS),
                            dcc.Dropdown(id="quarter-dd", options=[{"label": "All", "value": "All"}], value="All", clearable=False, style=DD_STYLE),
                            dbc.Label("Month", className=LABEL_CLASS),
                            dcc.Dropdown(id="month-dd", options=[{"label": "All", "value": "All"}], value="All", clearable=False, style=DD_STYLE),
                            dbc.Label("Region", className=LABEL_CLASS),
                            dcc.Dropdown(id="region-dd", options=region_options, value=[], multi=True, placeholder="All regions", style=DD_STYLE),
                            dbc.Label("Category", className=LABEL_CLASS),
                            dcc.Dropdown(id="category-dd", options=category_options, value=[], multi=True, placeholder="All categories", style=DD_STYLE),
                            dbc.Label("Segment", className=LABEL_CLASS),
                            dcc.Dropdown(id="segment-dd", options=segment_options, value=[], multi=True, placeholder="All segments", style=DD_STYLE),
                            html.Hr(className="my-3"),
                            html.H6("ML Models", className="mb-2"),
                            dbc.Label("Model", className="mb-1"),
                            dcc.Dropdown(
                                id="ml-model",
                                options=[{"label": m, "value": m} for m in model_options],
                                value="Random Forest",
                                clearable=False,
                                style=DD_STYLE,
                            ),
                            dbc.Label("Test size", className="mt-2 mb-1"),
                            dcc.Slider(
                                id="ml-test-size",
                                min=0.1, max=0.4, step=0.05, value=0.2,
                                marks={0.1: "10%", 0.2: "20%", 0.3: "30%", 0.4: "40%"},
                            ),
                            dbc.Button("Train / Refresh", id="ml-train", color="primary", className="mt-2", n_clicks=0),
                            html.Small("Uses current filters (incl. preset) as training data.", className="text-muted d-block mt-2"),
                        ],
                        className="filters-border",
                    ),
                    width=2,
                    className="p-2 mt-2",
                ),
                dbc.Col(
                    [
                        html.H3("Office Supplies Sales Dashboard", className="mt-3"),
                        dcc.Tabs(
                            id="tabs",
                            value="tab-dashboard",
                            className="custom-tabs",
                            children=[
                                dcc.Tab(label="Dashboard", value="tab-dashboard", className="tab", selected_className="tab--selected"),
                                dcc.Tab(label="Distribution", value="tab-distribution", className="tab", selected_className="tab--selected"),
                                dcc.Tab(label="Classification", value="tab-classification", className="tab", selected_className="tab--selected"),
                                dcc.Tab(label="Models", value="tab-models", className="tab", selected_className="tab--selected"),
                                dcc.Tab(label="Clustering", value="tab-clustering", className="tab", selected_className="tab--selected"),
                                dcc.Tab(label="Raw Data", value="tab-raw-data", className="tab", selected_className="tab--selected"),
                            ],
                        ),
                        html.Div(id="tab-content", className="mt-2"),
                    ],
                    width=10,
                    className="p-3",
                ),
            ]
        ),
    ],
)

# Cascading dropdowns
@app.callback(Output("quarter-dd", "options"), Output("quarter-dd", "value"), Input("year-dd", "value"))
def update_quarters(selected_year):
    d = df if selected_year in (None, "All") else df[df["Year"] == int(selected_year)]
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    present = [q for q in quarters if q in d["Quarter"].dropna().unique()]
    options = [{"label": "All", "value": "All"}] + [{"label": q, "value": q} for q in present]
    return options, "All"


@app.callback(Output("month-dd", "options"), Output("month-dd", "value"), Input("year-dd", "value"), Input("quarter-dd", "value"))
def update_months(selected_year, selected_quarter):
    d = df
    if selected_year not in (None, "All"):
        d = d[d["Year"] == int(selected_year)]
    if selected_quarter not in (None, "All"):
        d = d[d["Quarter"] == selected_quarter]
    month_map = d[["MonthNum", "Month"]].dropna().drop_duplicates().sort_values("MonthNum")
    options = [{"label": "All", "value": "All"}] + [{"label": row["Month"], "value": str(int(row["MonthNum"]))} for _, row in month_map.iterrows()]
    return options, "All"


# Train ML model (stores results; model object not stored)
@app.callback(
    Output("ml-store", "data"),
    Input("ml-train", "n_clicks"),
    State("ml-model", "value"),
    State("ml-test-size", "value"),
    State("year-dd", "value"),
    State("quarter-dd", "value"),
    State("month-dd", "value"),
    State("date-preset", "value"),
    State("region-dd", "value"),
    State("category-dd", "value"),
    State("segment-dd", "value"),
    prevent_initial_call=True,
)
def train_model(n_clicks, model_name, test_size, year, quarter, month, date_preset, regions, categories, segments):
    data = apply_filters(df, year, quarter, month, date_preset, regions, categories, segments).copy()
    if data.empty or "Sales" not in data.columns:
        return {"error": "No data available for training with current filters."}

    data = data[np.isfinite(data["Sales"].astype(float))].copy()
    if data.shape[0] < 200:
        return {"error": f"Not enough rows to train reliably (found {data.shape[0]}). Broaden filters."}

    y = data["Sales"].astype(float)
    feature_cols = [c for c in ["Ship Mode", "Segment", "Region", "State", "Category", "Sub-Category", "Product ID", "Postal Code"] if c in data.columns]
    X = data[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=0)

    pipe = build_model_pipeline(model_name)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    payload = {
        "model": model_name,
        "rows": int(data.shape[0]),
        "features": feature_cols,
        "metrics": {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        },
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }

    try:
        model = pipe.named_steps["model"]
        feat_names = get_feature_names(pipe)
        if model_name == "Random Forest" and hasattr(model, "feature_importances_") and feat_names:
            imp = model.feature_importances_
            top_idx = np.argsort(imp)[-20:][::-1]
            payload["importance"] = {"type": "rf", "names": [feat_names[i] for i in top_idx], "values": [float(imp[i]) for i in top_idx]}
        elif model_name in ("Linear Regression", "Ridge") and hasattr(model, "coef_") and feat_names:
            coef = np.asarray(model.coef_).ravel()
            top_idx = np.argsort(np.abs(coef))[-20:][::-1]
            payload["importance"] = {"type": "linear", "names": [feat_names[i] for i in top_idx], "values": [float(coef[i]) for i in top_idx]}
    except Exception:
        pass

    return payload


# Render tabs
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("year-dd", "value"),
    Input("quarter-dd", "value"),
    Input("month-dd", "value"),
    Input("date-preset", "value"),
    Input("region-dd", "value"),
    Input("category-dd", "value"),
    Input("segment-dd", "value"),
    Input("ml-store", "data"),
)
def render_tab(tab, year, quarter, month, date_preset, regions, categories, segments, ml_store):
    filtered = apply_filters(df, year, quarter, month, date_preset, regions, categories, segments)
    template = "plotly_white"

    # -------- Dashboard tab --------
    if tab == "tab-dashboard":
        kpi_base = apply_filters(df, "All", "All", "All", "ALL_TIME", regions, categories, segments)
        k = compute_kpis(kpi_base, filtered)
        kpis = dbc.Row(
            [
                dbc.Col(kpi_card("Total Sales", safe_currency(k["total_sales"])), md=4, lg=2),
                dbc.Col(kpi_card("Customer Count", safe_int(k["customer_count"])), md=4, lg=2),
                dbc.Col(kpi_card("Order Count", safe_int(k["order_count"])), md=4, lg=2),
                dbc.Col(kpi_card("YTD Sales", safe_currency(k["ytd_sales"])), md=4, lg=2),
                dbc.Col(kpi_card("LYTD Sales", safe_currency(k["lytd_sales"])), md=4, lg=2),
                dbc.Col(kpi_card("YoY %", safe_pct(k["yoy_pct"])), md=4, lg=2),
            ],
            className="g-2 mt-2",
        )

        hist_fc, fc_df = monthly_sales_forecast_series(filtered, horizon_months=6)
        if hist_fc is None or hist_fc.empty:
            fig_line = go.Figure()
            fig_line.update_layout(
                template=template,
                title="Sales by Month",
                margin=dict(l=20, r=20, t=60, b=20),
                annotations=[dict(text="No data for selected filters.", x=0.5, y=0.5, showarrow=False)],
            )
        else:
            fig_line = px.line(hist_fc, x="ds", y="y", markers=True, title="Sales by Month")
            fig_line.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))
            fig_line.update_xaxes(title_text="Month", tickformat="%Y-%m")
            fig_line.update_yaxes(title_text="Sales")
            fig_line = add_linear_trendline(fig_line, hist_fc["ds"], hist_fc["y"], name="Trend (Linear)")
            if fc_df is not None and not fc_df.empty:
                fig_line.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"], mode="lines+markers", name="Forecast"))
                fig_line.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig_line.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat_lower"], mode="lines", fill="tonexty", line=dict(width=0), showlegend=False, hoverinfo="skip"))

        if not filtered.empty and all(c in filtered.columns for c in ["Region", "Category", "Sales"]):
            sun_reg_cat = filtered.groupby(["Region", "Category"], as_index=False)["Sales"].sum()
        else:
            sun_reg_cat = pd.DataFrame({"Region": [], "Category": [], "Sales": []})
        fig_reg_cat = px.sunburst(sun_reg_cat, path=["Region", "Category"], values="Sales", title="Sales by Region / Category")
        fig_reg_cat.update_layout(template=template, margin=dict(l=10, r=10, t=60, b=20))

        if not filtered.empty and all(c in filtered.columns for c in ["Category", "Sub-Category", "Sales"]):
            sun_cat = filtered.groupby(["Category", "Sub-Category"], as_index=False)["Sales"].sum()
        else:
            sun_cat = pd.DataFrame({"Category": [], "Sub-Category": [], "Sales": []})
        fig_cat = px.sunburst(sun_cat, path=["Category", "Sub-Category"], values="Sales", title="Sales by Category / Sub-Category")
        fig_cat.update_layout(template=template, margin=dict(l=10, r=10, t=60, b=20))

        if not filtered.empty and all(c in filtered.columns for c in ["Region", "State", "Sales"]):
            sun_geo = filtered.groupby(["Region", "State"], as_index=False)["Sales"].sum()
        else:
            sun_geo = pd.DataFrame({"Region": [], "State": [], "Sales": []})
        fig_geo = px.sunburst(sun_geo, path=["Region", "State"], values="Sales", title="Sales by Region / State")
        fig_geo.update_layout(template=template, margin=dict(l=10, r=10, t=60, b=20))

        if not filtered.empty and all(c in filtered.columns for c in ["Region", "Segment", "Sales"]):
            sun_reg_seg = filtered.groupby(["Region", "Segment"], as_index=False)["Sales"].sum()
        else:
            sun_reg_seg = pd.DataFrame({"Region": [], "Segment": [], "Sales": []})
        fig_reg_seg = px.sunburst(sun_reg_seg, path=["Region", "Segment"], values="Sales", title="Sales by Region / Segment")
        fig_reg_seg.update_layout(template=template, margin=dict(l=10, r=10, t=60, b=20))

        return html.Div(
            [
                kpis,
                dbc.Row(
                    [
                        dbc.Col(html.Div(dcc.Graph(figure=fig_line), className="chart-border"), md=9),
                        dbc.Col(html.Div(dcc.Graph(figure=fig_reg_cat), className="chart-border"), md=3),
                    ],
                    className="mt-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(dcc.Graph(figure=fig_cat), className="chart-border"), md=4),
                        dbc.Col(html.Div(dcc.Graph(figure=fig_geo), className="chart-border"), md=4),
                        dbc.Col(html.Div(dcc.Graph(figure=fig_reg_seg), className="chart-border"), md=4),
                    ],
                    className="mt-2",
                ),
            ]
        )

    # -------- Distribution tab --------
    if tab == "tab-distribution":
        sales_df = pd.DataFrame()
        if "Sales" in filtered.columns:
            sales_df = filtered.copy()
            sales_df["Sales"] = pd.to_numeric(sales_df["Sales"], errors="coerce")
            sales_df = sales_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Sales"])

        if sales_df.empty:
            fig_hist = go.Figure()
            fig_hist.update_layout(
                template=template,
                title="Sales Distribution Histogram",
                margin=dict(l=20, r=20, t=60, b=20),
                annotations=[dict(text="No sales data for selected filters.", x=0.5, y=0.5, showarrow=False)],
            )

            fig_box = go.Figure()
            fig_box.update_layout(
                template=template,
                title="Sales Distribution Box Plot",
                margin=dict(l=20, r=20, t=60, b=20),
                annotations=[dict(text="No sales data for selected filters.", x=0.5, y=0.5, showarrow=False)],
            )

            fig_kde = go.Figure()
            fig_kde.update_layout(
                template=template,
                title="Sales Density Estimate",
                margin=dict(l=20, r=20, t=60, b=20),
                annotations=[dict(text="No sales data for selected filters.", x=0.5, y=0.5, showarrow=False)],
            )

            fig_qq = go.Figure()
            fig_qq.update_layout(
                template=template,
                title="Sales Q-Q Plot",
                margin=dict(l=20, r=20, t=60, b=20),
                annotations=[dict(text="No sales data for selected filters.", x=0.5, y=0.5, showarrow=False)],
            )
        else:
            nbins = max(10, min(60, int(np.sqrt(len(sales_df)))))
            fig_hist = px.histogram(
                sales_df,
                x="Sales",
                nbins=nbins,
                marginal="rug",
                title="Sales Distribution Histogram",
                labels={"Sales": "Sales"},
            )
            fig_hist.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))
            fig_hist.update_yaxes(title_text="Order Count")

            group_col = None
            for candidate in ["Category", "Region", "Segment"]:
                if candidate in sales_df.columns and sales_df[candidate].notna().any():
                    group_col = candidate
                    break

            if group_col:
                box_df = sales_df.dropna(subset=[group_col]).copy()
                fig_box = px.box(
                    box_df,
                    x=group_col,
                    y="Sales",
                    points="outliers",
                    title=f"Sales Box & Whisker by {group_col}",
                    labels={group_col: group_col, "Sales": "Sales"},
                )
                fig_box.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))
                fig_box.update_xaxes(tickangle=-30)
            else:
                fig_box = px.box(
                    sales_df,
                    y="Sales",
                    points="outliers",
                    title="Sales Box & Whisker Plot",
                    labels={"Sales": "Sales"},
                )
                fig_box.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))

            fig_kde = ff.create_distplot(
                [sales_df["Sales"].tolist()],
                ["Sales"],
                show_hist=False,
                show_rug=False,
            )
            fig_kde.update_layout(
                template=template,
                title="Sales Density Estimate",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            fig_kde.update_xaxes(title_text="Sales")
            fig_kde.update_yaxes(title_text="Density")

            qq_x, qq_y = stats.probplot(sales_df["Sales"].to_numpy(dtype=float), dist="norm", fit=False)
            qq_fit = stats.probplot(sales_df["Sales"].to_numpy(dtype=float), dist="norm", fit=True)
            slope, intercept, _ = qq_fit[1]
            qq_line_x = np.asarray(qq_x, dtype=float)
            qq_line_y = slope * qq_line_x + intercept

            fig_qq = go.Figure()
            fig_qq.add_trace(
                go.Scatter(
                    x=qq_x,
                    y=qq_y,
                    mode="markers",
                    name="Sample Quantiles",
                )
            )
            fig_qq.add_trace(
                go.Scatter(
                    x=qq_line_x,
                    y=qq_line_y,
                    mode="lines",
                    name="Reference Line",
                    line=dict(dash="dash"),
                )
            )
            fig_qq.update_layout(
                template=template,
                title="Sales Q-Q Plot",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            fig_qq.update_xaxes(title_text="Theoretical Quantiles")
            fig_qq.update_yaxes(title_text="Sample Quantiles")

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Div(dcc.Graph(figure=fig_hist), className="chart-border"), md=7),
                        dbc.Col(html.Div(dcc.Graph(figure=fig_box), className="chart-border"), md=5),
                    ],
                    className="mt-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(dcc.Graph(figure=fig_kde), className="chart-border"), md=6),
                        dbc.Col(html.Div(dcc.Graph(figure=fig_qq), className="chart-border"), md=6),
                    ],
                    className="mt-2",
                )
            ]
        )

    # -------- Classification tab --------
    if tab == "tab-classification":
        cls = compute_classification_artifacts(filtered)
        if "error" in cls:
            return dbc.Alert(cls["error"], color="info", className="mt-2")

        y_test = cls["y_test"]
        y_score = cls["y_score"]
        y_pred = cls["y_pred"]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = float(auc(fpr, tpr))
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        avg_precision = float(average_precision_score(y_test, y_score))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
        roc_fig.update_layout(template=template, title="ROC Curve", margin=dict(l=20, r=20, t=60, b=20))
        roc_fig.update_xaxes(title_text="False Positive Rate")
        roc_fig.update_yaxes(title_text="True Positive Rate")

        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AP={avg_precision:.3f})"))
        pr_fig.update_layout(template=template, title="Precision-Recall Curve", margin=dict(l=20, r=20, t=60, b=20))
        pr_fig.update_xaxes(title_text="Recall")
        pr_fig.update_yaxes(title_text="Precision")

        cm_fig = px.imshow(
            cm,
            x=["Pred: Low", "Pred: High"],
            y=["Actual: Low", "Actual: High"],
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        cm_fig.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))
        cm_fig.update_xaxes(side="top")

        sankey_labels = ["Actual: Low", "Actual: High", "Pred: Low", "Pred: High"]
        sankey_fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=18,
                        thickness=18,
                        label=sankey_labels,
                    ),
                    link=dict(
                        source=[0, 0, 1, 1],
                        target=[2, 3, 2, 3],
                        value=[
                            int(cm[0, 0]),
                            int(cm[0, 1]),
                            int(cm[1, 0]),
                            int(cm[1, 1]),
                        ],
                    ),
                )
            ]
        )
        sankey_fig.update_layout(template=template, title="Actual to Predicted Flow", margin=dict(l=20, r=20, t=60, b=20))

        accuracy = float((cm[0, 0] + cm[1, 1]) / max(1, cm.sum()))
        metrics_cards = dbc.Row(
            [
                dbc.Col(kpi_card("High-Value Cutoff", safe_currency(cls["threshold"])), md=4, lg=3),
                dbc.Col(kpi_card("Positive Rate", safe_pct(cls["positive_rate"])), md=4, lg=3),
                dbc.Col(kpi_card("ROC AUC", f"{roc_auc:.3f}"), md=4, lg=3),
                dbc.Col(kpi_card("Accuracy", safe_pct(accuracy)), md=4, lg=3),
            ],
            className="g-2 mt-2",
        )

        return html.Div(
            [
                metrics_cards,
                dbc.Row(
                    [
                        dbc.Col(html.Div(dcc.Graph(figure=roc_fig), className="chart-border"), md=6),
                        dbc.Col(html.Div(dcc.Graph(figure=pr_fig), className="chart-border"), md=6),
                    ],
                    className="mt-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(dcc.Graph(figure=cm_fig), className="chart-border"), md=5),
                        dbc.Col(html.Div(dcc.Graph(figure=sankey_fig), className="chart-border"), md=7),
                    ],
                    className="mt-2",
                ),
            ]
        )

    # -------- Raw Data tab --------
    if tab == "tab-raw-data":
        display_df = filtered.copy()
        for col in display_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
            display_df[col] = display_df[col].dt.strftime("%Y-%m-%d")
        display_df = display_df.replace({np.nan: None})

        return html.Div(
            [
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in display_df.columns],
                    data=display_df.to_dict("records"),
                    page_size=100,
                    page_action="native",
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "8px", "minWidth": "120px", "maxWidth": "240px"},
                    style_header={"fontWeight": "bold"},
                )
            ],
            className="mt-2",
        )

    # -------- Models tab (ONLY supervised ML evaluation) --------
    if tab == "tab-models":
        if not ml_store or "error" in ml_store:
            msg = (ml_store.get("error") if isinstance(ml_store, dict) else None) or "Click “Train / Refresh” to fit a model."
            return dbc.Alert(msg, color="info", className="mt-2")

        y_test = np.array(ml_store["y_test"], dtype=float)
        y_pred = np.array(ml_store["y_pred"], dtype=float)
        resid = y_test - y_pred

        metrics = ml_store["metrics"]
        metrics_cards = dbc.Row(
            [
                dbc.Col(kpi_card("Model", ml_store["model"]), md=4, lg=3),
                dbc.Col(kpi_card("R²", f"{metrics['r2']:.3f}"), md=4, lg=3),
                dbc.Col(kpi_card("MAE", safe_currency(metrics["mae"])), md=4, lg=3),
                dbc.Col(kpi_card("RMSE", safe_currency(metrics["rmse"])), md=4, lg=3),
            ],
            className="g-2 mt-2",
        )

        fig_scatter = px.scatter(
            x=y_test,
            y=y_pred,
            labels={"x": "Actual Sales", "y": "Predicted Sales"},
            title="Actual vs Predicted (Test Set)",
            opacity=0.65,
        )
        if y_test.size:
            mn, mx = float(np.min(y_test)), float(np.max(y_test))
            fig_scatter.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Ideal", line=dict(dash="dash")))
        fig_scatter.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20), height=340)

        fig_resid = px.histogram(x=resid, nbins=40, title="Residual Distribution (Actual - Predicted)", labels={"x": "Residual"})
        fig_resid.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20), height=340)

        if "importance" in ml_store:
            imp = ml_store["importance"]
            fig_imp = px.bar(
                x=imp["values"][::-1],
                y=imp["names"][::-1],
                orientation="h",
                title="Top Feature Importance" if imp["type"] == "rf" else "Top Coefficients (by |magnitude|)",
                labels={"x": "Importance" if imp["type"] == "rf" else "Coefficient", "y": "Feature"},
            )
            fig_imp.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20), height=440)
        else:
            fig_imp = go.Figure()
            fig_imp.update_layout(
                template=template,
                title="Feature importance not available for this run (or feature names could not be extracted).",
                margin=dict(l=20, r=20, t=60, b=20),
                height=440,
            )

        return html.Div(
            [
                metrics_cards,
                dbc.Row(
                    [
                        dbc.Col(html.Div(dcc.Graph(figure=fig_scatter), className="chart-border"), md=6),
                        dbc.Col(html.Div(dcc.Graph(figure=fig_resid), className="chart-border"), md=6),
                    ],
                    className="mt-2",
                ),
                dbc.Row([dbc.Col(html.Div(dcc.Graph(figure=fig_imp), className="chart-border"), md=12)], className="mt-2"),
            ]
        )

    # -------- Clustering tab (moved everything here) --------
    clus = build_cluster_df(filtered)

    # category clusters scatter
    if clus.empty:
        fig_cluster = go.Figure()
        fig_cluster.update_layout(
            template=template,
            title="Category Clusters (Region × Category)",
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=[dict(text="Not enough data/metrics to cluster for current filters.", x=0.5, y=0.5, showarrow=False)],
        )
    else:
        y_axis = "Profit" if "Profit" in clus.columns else ("OrderCount" if "OrderCount" in clus.columns else "CustomerCount")
        fig_cluster = px.scatter(
            clus,
            x="Sales",
            y=y_axis,
            color=clus["Cluster"].astype(str),
            hover_data={"Region": True, "Category": True, "Sales": ":,.2f", y_axis: True, "Cluster": True},
            title=f"Category Clusters (Region × Category) — x=Sales, y={y_axis}",
        )
        fig_cluster.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))

    # kmeans heatmap
    centers, z = build_kmeans_heatmap(clus) if not clus.empty else (None, None)
    if centers is None or z is None:
        fig_heat = go.Figure()
        fig_heat.update_layout(
            template=template,
            title="KMeans Cluster Heatmap (centroids)",
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=[dict(text="No clusters available to summarize.", x=0.5, y=0.5, showarrow=False)],
        )
    else:
        fig_heat = px.imshow(
            z.values,
            x=list(z.columns),
            y=[f"Cluster {i}" for i in z.index.tolist()],
            aspect="auto",
            title="KMeans Cluster Heatmap (centroids, z-scored by feature)",
            labels=dict(x="Feature", y="Cluster", color="z"),
        )
        fig_heat.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))

    # inertia curve
    inertia_df = clusters_vs_inertia(clus, k_min=2, k_max=10) if not clus.empty else None
    if inertia_df is None or inertia_df.empty:
        fig_inertia = go.Figure()
        fig_inertia.update_layout(
            template=template,
            title="Clusters vs Inertia (Elbow)",
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=[dict(text="Not enough points/features to compute an elbow curve.", x=0.5, y=0.5, showarrow=False)],
        )
    else:
        fig_inertia = px.line(
            inertia_df,
            x="k",
            y="inertia",
            markers=True,
            title="Clusters vs Inertia (Elbow Curve)",
            labels={"k": "# Clusters (k)", "inertia": "Inertia (within-cluster SSE)"},
        )
        fig_inertia.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))

    # silhouette curve
    silhouette_df = clusters_vs_silhouette(clus, k_min=2, k_max=10) if not clus.empty else None
    if silhouette_df is None or silhouette_df.empty:
        fig_silhouette = go.Figure()
        fig_silhouette.update_layout(
            template=template,
            title="Clusters vs Silhouette Score",
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=[dict(text="Not enough points/features to compute silhouette scores.", x=0.5, y=0.5, showarrow=False)],
        )
    else:
        fig_silhouette = px.line(
            silhouette_df,
            x="k",
            y="silhouette",
            markers=True,
            title="Clusters vs Silhouette Score",
            labels={"k": "# Clusters (k)", "silhouette": "Silhouette Score"},
        )
        fig_silhouette.update_layout(template=template, margin=dict(l=20, r=20, t=60, b=20))

    # dendrogram
    fig_dendo = build_dendrogram(clus, template)
    cluster_chart_height = 340
    dendrogram_chart_height = 340
    for fig in [fig_cluster, fig_heat, fig_inertia, fig_silhouette]:
        fig.update_layout(height=cluster_chart_height)
    fig_dendo.update_layout(height=dendrogram_chart_height)

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Div(dcc.Graph(figure=fig_cluster), className="chart-border"), md=4),
                    dbc.Col(html.Div(dcc.Graph(figure=fig_heat), className="chart-border"), md=4),
                    dbc.Col(html.Div(dcc.Graph(figure=fig_inertia), className="chart-border"), md=4),
                ],
                className="mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(dcc.Graph(figure=fig_silhouette), className="chart-border"), md=6),
                    dbc.Col(html.Div(dcc.Graph(figure=fig_dendo), className="chart-border"), md=6),
                ],
                className="mt-2",
            ),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
