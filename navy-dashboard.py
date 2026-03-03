from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, Input, Output, dcc, html, no_update
from dash import dash_table

DATA_PATH = "superstore.xlsx"

# ----- Theme -----
NAVY = "#0B1E3A"
NAVY_2 = "#0F2A4A"
CYAN = "#00E5FF"
CYAN_2 = "#B8F8FF"
BORDER = "#2E5C86"

BLUE_SHADES = [
    "#0A3D91", "#0B4BAE", "#0C5CCB", "#0D6EE8",
    "#1D7FFF", "#3D92FF", "#5FA6FF", "#7BB9FF",
    "#97CDFF", "#B3E1FF", "#D0F1FF", "#E6FBFF"
]


def shade_list(n: int) -> list[str]:
    if n <= 0:
        return []
    return (BLUE_SHADES * ((n // len(BLUE_SHADES)) + 1))[:n]


PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=NAVY,
        plot_bgcolor=NAVY,
        font=dict(color=CYAN, family="Inter, Segoe UI, Arial"),
        title=dict(font=dict(color=CYAN)),
        xaxis=dict(
            gridcolor="rgba(0,229,255,0.12)",
            zerolinecolor="rgba(0,229,255,0.20)",
            linecolor="rgba(0,229,255,0.25)",
            tickfont=dict(color=CYAN)
        ),
        yaxis=dict(
            gridcolor="rgba(0,229,255,0.12)",
            zerolinecolor="rgba(0,229,255,0.20)",
            linecolor="rgba(0,229,255,0.25)",
            tickfont=dict(color=CYAN)
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CYAN)),
        hoverlabel=dict(
            bgcolor=NAVY_2,
            bordercolor=BORDER,
            font=dict(color=CYAN_2),
        ),
        margin=dict(l=60, r=18, t=54, b=60),
    )
)

import plotly.io as pio  # noqa: E402
pio.templates["navy_cyan"] = PLOTLY_TEMPLATE
pio.templates.default = "navy_cyan"


# ----- Data -----
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df


df = load_data(DATA_PATH)

region_opts = sorted(df["Region"].dropna().unique().tolist())
category_opts = sorted(df["Category"].dropna().unique().tolist())
segment_opts = sorted(df["Segment"].dropna().unique().tolist())

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()
anchor_date = pd.to_datetime(df["Order Date"].max()).normalize()


def clamp_date(d: pd.Timestamp) -> pd.Timestamp:
    d = d.normalize()
    d = max(d, pd.to_datetime(min_date))
    d = min(d, pd.to_datetime(max_date))
    return d


def filter_df(
    df_: pd.DataFrame,
    start_date,
    end_date,
    regions: list[str],
    categories: list[str],
    segments: list[str],
) -> pd.DataFrame:
    out = df_.copy()
    out = out[
        (out["Order Date"].dt.date >= pd.to_datetime(start_date).date())
        & (out["Order Date"].dt.date <= pd.to_datetime(end_date).date())
    ]
    if regions:
        out = out[out["Region"].isin(regions)]
    if categories:
        out = out[out["Category"].isin(categories)]
    if segments:
        out = out[out["Segment"].isin(segments)]
    return out


def compute_ytd_metrics(base_df: pd.DataFrame, end_date) -> tuple[float, float, float | None]:
    end_dt = clamp_date(pd.to_datetime(end_date))
    ytd_start = clamp_date(pd.Timestamp(year=end_dt.year, month=1, day=1))
    ly_end = clamp_date(end_dt - pd.DateOffset(years=1))
    ly_start = clamp_date(pd.Timestamp(year=ly_end.year, month=1, day=1))

    ytd = base_df[(base_df["Order Date"] >= ytd_start) & (base_df["Order Date"] <= end_dt)]["Sales"].sum()
    lytd = base_df[(base_df["Order Date"] >= ly_start) & (base_df["Order Date"] <= ly_end)]["Sales"].sum()

    yoy = None
    if lytd and float(lytd) != 0.0:
        yoy = (float(ytd) - float(lytd)) / float(lytd)
    return float(ytd), float(lytd), yoy


def add_linear_trendline(fig: go.Figure, x_vals: list, y_vals: list, name: str = "Trend") -> go.Figure:
    if len(x_vals) < 2:
        return fig
    x_idx = np.arange(len(x_vals), dtype=float)
    y_arr = np.array(y_vals, dtype=float)
    if np.all(np.isnan(y_arr)):
        return fig
    m, b = np.polyfit(x_idx, y_arr, 1)
    y_hat = m * x_idx + b
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_hat,
            mode="lines",
            name=name,
            line=dict(width=2, dash="dash", color=CYAN_2),
            hoverinfo="skip",
        )
    )
    return fig


# ----- App -----
app = Dash(__name__, title="Superstore Dashboard (Navy/Cyan)")

app.index_string = f"""
<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>
      :root {{
        --navy: {NAVY};
        --navy2: {NAVY_2};
        --cyan: {CYAN};
        --cyan2: {CYAN_2};
        --border: {BORDER};
      }}
      html, body {{
        background: var(--navy);
        color: var(--cyan);
        margin: 0;
        font-family: Inter, Segoe UI, Arial, sans-serif;
      }}
      .app-container {{
        max-width: 1480px;
        margin: 0 auto;
        padding: 14px 14px 24px 14px;
      }}
      .layout {{
        display: grid;
        grid-template-columns: 340px 1fr;
        gap: 12px;
        align-items: start;
      }}
      @media (max-width: 1100px) {{
        .layout {{ grid-template-columns: 1fr; }}
      }}
      .panel {{
        position: sticky;
        top: 12px;
      }}
      .card {{
        background: linear-gradient(180deg, rgba(15,42,74,0.95), rgba(11,30,58,0.95));
        border: 1px solid rgba(46,92,134,0.65);
        border-radius: 14px;
        padding: 14px 14px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      }}
      .title {{
        margin-bottom: 10px;
      }}
      .title h2 {{
        margin: 0;
        font-weight: 900;
        letter-spacing: 0.2px;
        color: var(--cyan);
        font-size: 22px;
      }}
      .subtitle {{
        color: rgba(0,229,255,0.75);
        margin-top: 4px;
        font-size: 12px;
        line-height: 1.25;
      }}

      .label {{
        font-size: 12px;
        color: rgba(0,229,255,0.85);
        margin: 0 0 6px 2px;
      }}

      /* DatePicker */
      .DateInput_input {{
        background: var(--navy2) !important;
        color: var(--cyan) !important;
        border: 1px solid rgba(46,92,134,0.65) !important;
      }}
      .DateRangePickerInput {{
        background: transparent !important;
      }}
      .DateRangePickerInput_arrow_svg {{
        fill: var(--cyan) !important;
      }}

      /* react-select (dcc.Dropdown) */
      .Select-control {{
        background: var(--navy2) !important;
        border: 1px solid rgba(46,92,134,0.65) !important;
        color: var(--cyan) !important;
        border-radius: 10px !important;
      }}
      .Select-placeholder, .Select--single > .Select-control .Select-value {{
        color: rgba(0,229,255,0.70) !important;
      }}
      .Select-menu-outer {{
        background: var(--navy2) !important;
        border: 1px solid rgba(46,92,134,0.65) !important;
        color: var(--cyan) !important;
        border-radius: 10px !important;
      }}
      .Select-option {{
        background: var(--navy2) !important;
        color: var(--cyan) !important;
      }}
      .Select-option.is-focused {{
        background: rgba(0,229,255,0.12) !important;
      }}
      .Select-option.is-selected {{
        background: rgba(0,229,255,0.20) !important;
      }}

      /* Radio items (date presets) */
      .preset-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}
      .preset-row label {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        cursor: pointer;
        color: rgba(0,229,255,0.85);
        font-size: 12px;
        padding: 6px 10px;
        border: 1px solid rgba(46,92,134,0.55);
        border-radius: 999px;
        background: rgba(15,42,74,0.55);
      }}
      .preset-row input {{
        accent-color: var(--cyan);
      }}

      /* Main content grid */
      .main {{
        display: flex;
        flex-direction: column;
        gap: 12px;
      }}
      .grid {{
        display: grid;
        gap: 12px;
      }}
      .kpis {{
        grid-template-columns: repeat(7, 1fr);
      }}
      .row-2fr-1fr {{
        grid-template-columns: 2fr 1fr;
        align-items: stretch;
      }}
      .row-1fr-1fr {{
        grid-template-columns: 1fr 1fr;
      }}
      .row-1fr {{
        grid-template-columns: 1fr;
      }}
      @media (max-width: 1250px) {{
        .kpis {{ grid-template-columns: repeat(2, 1fr); }}
        .row-2fr-1fr {{ grid-template-columns: 1fr; }}
        .row-1fr-1fr {{ grid-template-columns: 1fr; }}
      }}

      /* KPI centered */
      .kpi {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 92px;
      }}
      .kpi-title {{
        color: rgba(0,229,255,0.75);
        font-size: 12px;
        margin-bottom: 6px;
      }}
      .kpi-value {{
        font-size: 24px;
        font-weight: 900;
        line-height: 1.1;
        color: var(--cyan);
      }}
      .kpi-sub {{
        margin-top: 6px;
        font-size: 12px;
        color: rgba(0,229,255,0.65);
      }}

      /* DataTable */
      .dash-table-container .dash-spreadsheet-container {{
        background: transparent !important;
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


def kpi_block(title: str, value: str, subtitle: str | None = None) -> html.Div:
    return html.Div(
        className="card kpi",
        children=[
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value"),
            html.Div(subtitle, className="kpi-sub") if subtitle else None,
        ],
    )


def make_table(table_id: str) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=table_id,
        columns=[],
        data=[],
        page_size=10,
        sort_action="native",
        style_header={
            "backgroundColor": NAVY_2,
            "color": CYAN,
            "border": f"1px solid {BORDER}",
            "fontWeight": "900",
        },
        style_cell={
            "backgroundColor": NAVY,
            "color": CYAN,
            "border": f"1px solid {BORDER}",
            "padding": "8px",
            "fontFamily": "Inter, Segoe UI, Arial",
            "fontSize": "12px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_table={
            "backgroundColor": "transparent",
            "borderRadius": "12px",
            "overflow": "hidden",
        },
    )


app.layout = html.Div(
    className="app-container",
    children=[
        html.Div(
            className="layout",
            children=[
                # Left filter panel
                html.Div(
                    className="panel",
                    children=[
                        html.Div(
                            className="card",
                            children=[
                                html.Div(
                                    className="title",
                                    children=[
                                        html.H2("Filters"),
                                        html.Div(
                                            f"Presets anchored to latest Order Date in data ({anchor_date.date()})",
                                            className="subtitle",
                                        ),
                                    ],
                                ),

                                html.Div("Date Preset", className="label"),
                                dcc.RadioItems(
                                    id="date-preset",
                                    className="preset-row",
                                    options=[
                                        {"label": "Custom", "value": "custom"},
                                        {"label": "Today", "value": "today"},
                                        {"label": "Last Week", "value": "last_week"},
                                        {"label": "Last Month", "value": "last_month"},
                                        {"label": "Last Quarter", "value": "last_quarter"},
                                        {"label": "Last 6 Months", "value": "last_6_months"},
                                        {"label": "Last Year", "value": "last_year"},
                                        {"label": "Last 2 Years", "value": "last_2_years"},
                                        {"label": "Last 3 Years", "value": "last_3_years"},
                                        {"label": "All Time", "value": "all_time"},
                                    ],
                                    value="last_year",
                                    inline=True,
                                ),
                                html.Div(style={"height": "10px"}),

                                html.Div("Order Date Range", className="label"),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    min_date_allowed=min_date,
                                    max_date_allowed=max_date,
                                    start_date=min_date,
                                    end_date=max_date,
                                    display_format="YYYY-MM-DD",
                                    clearable=False,
                                ),
                                html.Div(style={"height": "12px"}),

                                html.Div("Region", className="label"),
                                dcc.Dropdown(
                                    id="region-dd",
                                    options=[{"label": r, "value": r} for r in region_opts],
                                    value=[],
                                    multi=True,
                                    placeholder="All regions",
                                ),
                                html.Div(style={"height": "10px"}),

                                html.Div("Category", className="label"),
                                dcc.Dropdown(
                                    id="category-dd",
                                    options=[{"label": c, "value": c} for c in category_opts],
                                    value=[],
                                    multi=True,
                                    placeholder="All categories",
                                ),
                                html.Div(style={"height": "10px"}),

                                html.Div("Segment", className="label"),
                                dcc.Dropdown(
                                    id="segment-dd",
                                    options=[{"label": s, "value": s} for s in segment_opts],
                                    value=[],
                                    multi=True,
                                    placeholder="All segments",
                                ),
                            ],
                        )
                    ],
                ),

                # Main content
                html.Div(
                    className="main",
                    children=[
                        html.Div(
                            className="card",
                            children=[
                                html.Div(
                                    style={"fontWeight": "900", "fontSize": "20px", "marginBottom": "6px"},
                                    children="Superstore Dashboard",
                                ),
                                html.Div(
                                    className="subtitle",
                                    children="Navy background • Cyan text • Trendline • Top states/customers/products • YTD & YoY KPIs",
                                ),
                            ],
                        ),

                        # KPI row
                        html.Div(
                            className="grid kpis",
                            children=[
                                html.Div(id="kpi-sales"),
                                html.Div(id="kpi-profit"),
                                html.Div(id="kpi-orders"),
                                html.Div(id="kpi-customers"),
                                html.Div(id="kpi-ytd"),
                                html.Div(id="kpi-lytd"),
                                html.Div(id="kpi-yoy"),
                            ],
                        ),

                        # Charts row 1
                        html.Div(
                            className="grid row-2fr-1fr",
                            children=[
                                html.Div(className="card", children=[dcc.Graph(id="fig-month", config={"displayModeBar": False})]),
                                html.Div(className="card", children=[dcc.Graph(id="fig-region", config={"displayModeBar": False})]),
                            ],
                        ),

                        # Charts row 2
                        html.Div(
                            className="grid row-1fr-1fr",
                            children=[
                                html.Div(className="card", children=[dcc.Graph(id="fig-category", config={"displayModeBar": False})]),
                                html.Div(className="card", children=[dcc.Graph(id="fig-subcat-top", config={"displayModeBar": False})]),
                            ],
                        ),

                        # Charts row 3
                        html.Div(
                            className="grid row-1fr-1fr",
                            children=[
                                html.Div(className="card", children=[dcc.Graph(id="fig-states-top", config={"displayModeBar": False})]),
                                html.Div(
                                    className="card",
                                    children=[
                                        html.Div("Top 10 Customers (by Sales)", className="label"),
                                        make_table("tbl-top-customers"),
                                    ],
                                ),
                            ],
                        ),

                        # NEW: Pivot-style table for Top 10 Products
                        html.Div(
                            className="grid row-1fr",
                            children=[
                                html.Div(
                                    className="card",
                                    children=[
                                        html.Div("Top 10 Selling Products (Pivot)", className="label"),
                                        html.Div(
                                            className="subtitle",
                                            children="Aggregated by Product Name (Sales, Profit, Quantity, Orders)",
                                        ),
                                        make_table("tbl-top-products"),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
            ],
        )
    ],
)


# ---- Preset -> DatePickerRange ----
@app.callback(
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("date-preset", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def apply_date_preset(preset, current_start, current_end):
    if preset == "custom":
        return current_start, current_end

    end = anchor_date

    if preset == "today":
        start = end
    elif preset == "last_week":
        start = end - pd.Timedelta(days=6)
    elif preset == "last_month":
        start = end - pd.Timedelta(days=29)
    elif preset == "last_quarter":
        start = end - pd.DateOffset(months=3)
    elif preset == "last_6_months":
        start = end - pd.DateOffset(months=6)
    elif preset == "last_year":
        start = end - pd.DateOffset(years=1)
    elif preset == "last_2_years":
        start = end - pd.DateOffset(years=2)
    elif preset == "last_3_years":
        start = end - pd.DateOffset(years=3)
    elif preset == "all_time":
        start = pd.to_datetime(min_date)
    else:
        return no_update, no_update

    start = clamp_date(pd.to_datetime(start)).date()
    end = clamp_date(pd.to_datetime(end)).date()
    return start, end


# ---- Main dashboard ----
@app.callback(
    Output("kpi-sales", "children"),
    Output("kpi-profit", "children"),
    Output("kpi-orders", "children"),
    Output("kpi-customers", "children"),
    Output("kpi-ytd", "children"),
    Output("kpi-lytd", "children"),
    Output("kpi-yoy", "children"),
    Output("fig-month", "figure"),
    Output("fig-region", "figure"),
    Output("fig-category", "figure"),
    Output("fig-subcat-top", "figure"),
    Output("fig-states-top", "figure"),
    Output("tbl-top-customers", "columns"),
    Output("tbl-top-customers", "data"),
    Output("tbl-top-products", "columns"),
    Output("tbl-top-products", "data"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("region-dd", "value"),
    Input("category-dd", "value"),
    Input("segment-dd", "value"),
)
def update_dashboard(start_date, end_date, regions, categories, segments):
    dff = filter_df(df, start_date, end_date, regions or [], categories or [], segments or [])

    total_sales = float(dff["Sales"].sum())
    total_profit = float(dff["Profit"].sum())
    orders = int(dff["Order ID"].nunique())
    customers = int(dff["Customer ID"].nunique())

    # YTD / LYTD / YoY computed on dimension filters and selected end_date
    base_dim_df = df.copy()
    if regions:
        base_dim_df = base_dim_df[base_dim_df["Region"].isin(regions)]
    if categories:
        base_dim_df = base_dim_df[base_dim_df["Category"].isin(categories)]
    if segments:
        base_dim_df = base_dim_df[base_dim_df["Segment"].isin(segments)]
    ytd, lytd, yoy = compute_ytd_metrics(base_dim_df, end_date)

    kpi_sales = kpi_block("Total Sales", f"${total_sales:,.0f}", subtitle=f"Rows: {len(dff):,}")
    kpi_profit = kpi_block("Total Profit", f"${total_profit:,.0f}")
    kpi_orders = kpi_block("Orders", f"{orders:,}")
    kpi_customers = kpi_block("Customers", f"{customers:,}")
    kpi_ytd = kpi_block("YTD Sales", f"${ytd:,.0f}", subtitle=f"Through {pd.to_datetime(end_date).date()}")
    kpi_lytd = kpi_block("LYTD Sales", f"${lytd:,.0f}", subtitle="Same period last year")
    yoy_text = "—" if yoy is None else f"{yoy*100:,.1f}%"
    kpi_yoy = kpi_block("YoY Sales %", yoy_text)

    # Line: Sales by Month (cyan) + trendline
    monthly = dff.groupby("Order Month", as_index=False).agg(Sales=("Sales", "sum")).sort_values("Order Month")
    fig_month = px.line(
        monthly,
        x="Order Month",
        y="Sales",
        markers=True,
        title="Sales by Month (Cyan Line + Trendline)",
        template="navy_cyan",
    )
    fig_month.update_traces(
        line=dict(width=3, color=CYAN),
        marker=dict(color=CYAN),
        selector=dict(mode="lines+markers"),
    )
    fig_month.update_layout(height=360, margin=dict(l=70, r=18, t=54, b=60))
    fig_month.update_yaxes(title_text="Sales ($)")
    fig_month.update_xaxes(title_text="")
    fig_month = add_linear_trendline(fig_month, monthly["Order Month"].tolist(), monthly["Sales"].tolist(), name="Trend")

    # Pie: Sales by Region (blue shades)
    by_region = dff.groupby("Region", as_index=False).agg(Sales=("Sales", "sum")).sort_values("Sales", ascending=False)
    fig_region = px.pie(
        by_region,
        names="Region",
        values="Sales",
        hole=0.45,
        title="Sales by Region",
        template="navy_cyan",
    )
    fig_region.update_traces(marker=dict(colors=shade_list(len(by_region))), textfont_color=CYAN)
    fig_region.update_layout(height=360, margin=dict(l=18, r=18, t=54, b=18))

    # Bar: Sales by Category (blue shades)
    by_cat = dff.groupby("Category", as_index=False).agg(Sales=("Sales", "sum")).sort_values("Sales", ascending=False)
    fig_category = px.bar(by_cat, x="Category", y="Sales", title="Sales by Category", template="navy_cyan")
    fig_category.update_traces(marker_color=shade_list(len(by_cat)))
    fig_category.update_layout(height=320, margin=dict(l=80, r=18, t=54, b=80))
    fig_category.update_yaxes(title_text="Sales ($)")
    fig_category.update_xaxes(title_text="")

    # Bar: Top 10 Sub-Categories
    by_sub = (
        dff.groupby("Sub-Category", as_index=False)
        .agg(Sales=("Sales", "sum"))
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    fig_sub = px.bar(
        by_sub.sort_values("Sales"),
        x="Sales",
        y="Sub-Category",
        orientation="h",
        title="Top 10 Sub-Categories by Sales",
        template="navy_cyan",
    )
    fig_sub.update_traces(marker_color=shade_list(len(by_sub))[::-1])
    fig_sub.update_layout(height=320, margin=dict(l=150, r=18, t=54, b=60))
    fig_sub.update_xaxes(title_text="Sales ($)")
    fig_sub.update_yaxes(title_text="")

    # Bar: Top 10 States
    by_state = dff.groupby("State", as_index=False).agg(Sales=("Sales", "sum")).sort_values("Sales", ascending=False).head(10)
    fig_states = px.bar(
        by_state.sort_values("Sales"),
        x="Sales",
        y="State",
        orientation="h",
        title="Top 10 States by Sales",
        template="navy_cyan",
    )
    fig_states.update_traces(marker_color=shade_list(len(by_state))[::-1])
    fig_states.update_layout(height=320, margin=dict(l=130, r=18, t=54, b=60))
    fig_states.update_xaxes(title_text="Sales ($)")
    fig_states.update_yaxes(title_text="")

    # Table: Top 10 Customers
    top_customers = (
        dff.groupby(["Customer Name"], as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"))
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    cust_tbl = top_customers.copy()
    cust_tbl["Sales"] = cust_tbl["Sales"].map(lambda v: f"${v:,.0f}")
    cust_tbl["Profit"] = cust_tbl["Profit"].map(lambda v: f"${v:,.0f}")
    cust_tbl["Orders"] = cust_tbl["Orders"].astype(int)

    cust_cols = [
        {"name": "Customer Name", "id": "Customer Name"},
        {"name": "Sales", "id": "Sales"},
        {"name": "Profit", "id": "Profit"},
        {"name": "Orders", "id": "Orders"},
    ]
    cust_data = cust_tbl.to_dict("records")

    # NEW Pivot Table: Top 10 Products (by Sales)
    # "Pivot" here means a grouped/aggregated table (Product Name as rows, measures as columns).
    top_products = (
        dff.pivot_table(
            index=["Product Name"],
            values=["Sales", "Profit", "Quantity"],
            aggfunc={"Sales": "sum", "Profit": "sum", "Quantity": "sum"},
        )
        .reset_index()
    )
    # add Orders count (nunique Order ID per product)
    orders_per_product = dff.groupby("Product Name", as_index=False).agg(Orders=("Order ID", "nunique"))
    top_products = top_products.merge(orders_per_product, on="Product Name", how="left")

    top_products = top_products.sort_values("Sales", ascending=False).head(10)

    prod_tbl = top_products.copy()
    prod_tbl["Sales"] = prod_tbl["Sales"].map(lambda v: f"${v:,.0f}")
    prod_tbl["Profit"] = prod_tbl["Profit"].map(lambda v: f"${v:,.0f}")
    prod_tbl["Quantity"] = prod_tbl["Quantity"].astype(int)
    prod_tbl["Orders"] = prod_tbl["Orders"].fillna(0).astype(int)

    prod_cols = [
        {"name": "Product Name", "id": "Product Name"},
        {"name": "Sales", "id": "Sales"},
        {"name": "Profit", "id": "Profit"},
        {"name": "Quantity", "id": "Quantity"},
        {"name": "Orders", "id": "Orders"},
    ]
    prod_data = prod_tbl.to_dict("records")

    return (
        kpi_sales,
        kpi_profit,
        kpi_orders,
        kpi_customers,
        kpi_ytd,
        kpi_lytd,
        kpi_yoy,
        fig_month,
        fig_region,
        fig_category,
        fig_sub,
        fig_states,
        cust_cols,
        cust_data,
        prod_cols,
        prod_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
