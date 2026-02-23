import pandas as pd
import numpy as np
from datetime import datetime

from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go


DATA_PATH = "OfficeSuppliesOrderData.xlsx"


# ----------------------------
# Data
# ----------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    df["Year"] = df["Order Date"].dt.year
    df["Quarter"] = "Q" + df["Order Date"].dt.quarter.astype("Int64").astype(str)
    df["Month"] = df["Order Date"].dt.month
    df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Lead Time (Days)"] = (df["Ship Date"] - df["Order Date"]).dt.days

    df = df.dropna(subset=["Order Date", "Sales"])
    return df


DF = load_data(DATA_PATH)


# ----------------------------
# Theme + Colors
# ----------------------------
# Light: white bg + navy text
LIGHT = {
    "bg": "#FFFFFF",
    "card": "#F6F8FC",
    "text": "#0A1F44",         # navy
    "muted": "#334155",
    "border": "1px solid rgba(10,31,68,0.25)",
    "plot_template": "plotly",
}

# Dark: navy bg + cyan text
DARK = {
    "bg": "#07162A",           # navy background
    "card": "#0B2240",         # slightly lighter navy
    "text": "#00FFFF",         # cyan text
    "muted": "#8FF7F7",
    "border": "1px solid rgba(0,255,255,0.45)",
    "plot_template": "plotly_dark",
}

CYAN = "#00FFFF"

BLUE_SHADES = [
    "#0B3D91",
    "#0F52BA",
    "#1565C0",
    "#1976D2",
    "#1E88E5",
    "#2196F3",
    "#42A5F5",
    "#64B5F6",
    "#90CAF9",
    "#BBDEFB",
]


def theme_tokens(theme_name: str):
    return DARK if theme_name == "dark" else LIGHT


def apply_theme(fig, tokens):
    fig.update_layout(
        template=tokens["plot_template"],
        paper_bgcolor=tokens["bg"],
        plot_bgcolor=tokens["bg"],
        font=dict(color=tokens["text"]),
        title=dict(font=dict(color=tokens["text"])),
        legend=dict(font=dict(color=tokens["text"])),
        margin=dict(l=20, r=20, t=55, b=20),
    )

    grid = "rgba(255,255,255,0.12)" if tokens["bg"] != "#FFFFFF" else "rgba(0,0,0,0.08)"
    fig.update_xaxes(showgrid=True, gridcolor=grid, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=grid, zeroline=False)
    return fig


def kpi_card(title: str, value: str, tokens):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "opacity": 0.9}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "700", "marginTop": "6px"}),
        ],
        style={
            "background": tokens["card"],
            "border": tokens["border"],
            "borderRadius": "12px",
            "padding": "14px 16px",
            "flex": "1",
            "minWidth": "180px",
        },
    )


# ----------------------------
# App
# ----------------------------
app = Dash(__name__)
app.title = "Office Supplies Sales Dashboard"

year_options = sorted([int(y) for y in DF["Year"].dropna().unique()])
quarter_options = ["Q1", "Q2", "Q3", "Q4"]
month_options = list(range(1, 13))
region_options = sorted(DF["Region"].dropna().unique())
category_options = sorted(DF["Category"].dropna().unique())

app.layout = html.Div(
    id="page",
    children=[
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Office Supplies — Interactive Sales Dashboard", style={"margin": 0}),
                        html.Div(
                            "Dash + Plotly | Theme toggle + filters + KPIs + drill-down",
                            style={"marginTop": "6px", "opacity": 0.9},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "2px"},
                ),
                html.Div(
                    [
                        html.Div("Theme", style={"fontSize": "12px", "opacity": 0.9, "marginBottom": "6px"}),
                        dcc.RadioItems(
                            id="theme",
                            options=[
                                {"label": " Dark (navy + cyan)", "value": "dark"},
                                {"label": " Light (white + navy)", "value": "light"},
                            ],
                            value="dark",
                            labelStyle={"display": "block", "cursor": "pointer", "marginBottom": "6px"},
                            inputStyle={"marginRight": "8px"},
                        ),
                    ],
                    style={"minWidth": "240px"},
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "gap": "16px"},
        ),

        html.Hr(style={"opacity": 0.25}),

        # Drilldown controls
        html.Div(
            [
                html.Button("Clear drilldown", id="clear_drill", n_clicks=0),
                html.Div(id="drill_label", style={"marginLeft": "12px", "fontSize": "12px", "opacity": 0.9}),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "10px"},
        ),

        html.Div(style={"height": "12px"}),

        # Filters
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Year", style={"fontSize": "12px", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="year",
                            className="navy-dropdown",
                            options=[{"label": str(y), "value": y} for y in year_options],
                            value=year_options,
                            multi=True,
                            clearable=True,
                        ),
                    ],
                    style={"flex": 1, "minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Div("Quarter", style={"fontSize": "12px", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="quarter",
                            className="navy-dropdown",
                            options=[{"label": q, "value": q} for q in quarter_options],
                            value=quarter_options,
                            multi=True,
                            clearable=True,
                        ),
                    ],
                    style={"flex": 1, "minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Div("Month", style={"fontSize": "12px", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="month",
                            className="navy-dropdown",
                            options=[{"label": datetime(2000, m, 1).strftime("%B"), "value": m} for m in month_options],
                            value=month_options,
                            multi=True,
                            clearable=True,
                        ),
                    ],
                    style={"flex": 1, "minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Div("Region (click pie to drill)", style={"fontSize": "12px", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="region",
                            className="navy-dropdown",
                            options=[{"label": r, "value": r} for r in region_options],
                            value=region_options,
                            multi=True,
                            clearable=True,
                        ),
                    ],
                    style={"flex": 1, "minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Div("Category (click bar to drill)", style={"fontSize": "12px", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="category",
                            className="navy-dropdown",
                            options=[{"label": c, "value": c} for c in category_options],
                            value=category_options,
                            multi=True,
                            clearable=True,
                        ),
                    ],
                    style={"flex": 1, "minWidth": "260px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
        ),

        html.Div(style={"height": "14px"}),

        # KPIs
        html.Div(id="kpis", style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}),

        html.Div(style={"height": "14px"}),

        # Charts
        html.Div(
            [
                html.Div([dcc.Graph(id="sales_trend")], style={"flex": 2, "minWidth": "520px"}),
                html.Div([dcc.Graph(id="sales_by_region")], style={"flex": 1, "minWidth": "420px"}),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
        ),

        html.Div(
            [
                html.Div([dcc.Graph(id="sales_by_category")], style={"flex": 1, "minWidth": "520px"}),
                html.Div([dcc.Graph(id="top_states")], style={"flex": 1, "minWidth": "520px"}),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginTop": "12px"},
        ),

        html.Div(
            [
                html.Div([dcc.Graph(id="top_subcats")], style={"flex": 1, "minWidth": "520px"}),
                html.Div([dcc.Graph(id="lead_time_dist")], style={"flex": 1, "minWidth": "520px"}),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginTop": "12px"},
        ),

        html.Div(style={"height": "14px"}),

        # Table
        html.Div(
            [
                html.H3("Top Customers (by Sales)", style={"marginTop": "6px"}),
                dash_table.DataTable(
                    id="top_customers_table",
                    page_size=10,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "padding": "10px",
                        "fontFamily": "Arial",
                        "fontSize": "12px",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                ),
            ],
            id="table_container",
            style={"marginTop": "8px"},
        ),

        html.Div(style={"height": "18px"}),

        html.Div(
            "Drill-down: click a Region slice or Category bar. Clear drilldown resets both.",
            style={"fontSize": "12px", "opacity": 0.85},
        ),
    ],
)


# ----------------------------
# Drilldown callback (click-to-filter)
# ----------------------------
@app.callback(
    Output("region", "value"),
    Output("category", "value"),
    Input("sales_by_region", "clickData"),
    Input("sales_by_category", "clickData"),
    Input("clear_drill", "n_clicks"),
    State("region", "value"),
    State("category", "value"),
    prevent_initial_call=True,
)
def drilldown(region_click, category_click, clear_clicks, current_regions, current_categories):
    from dash import callback_context

    current_regions = current_regions or region_options
    current_categories = current_categories or category_options

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

    if trigger == "clear_drill":
        return region_options, category_options

    if trigger == "sales_by_region" and region_click and region_click.get("points"):
        r = region_click["points"][0].get("label") or region_click["points"][0].get("x")
        if r:
            return [r], current_categories

    if trigger == "sales_by_category" and category_click and category_click.get("points"):
        c = category_click["points"][0].get("x") or category_click["points"][0].get("label")
        if c:
            return current_regions, [c]

    return current_regions, current_categories


# ----------------------------
# Main dashboard callback
# ----------------------------
@app.callback(
    Output("page", "style"),
    Output("clear_drill", "style"),
    Output("kpis", "children"),
    Output("drill_label", "children"),
    Output("sales_trend", "figure"),
    Output("sales_by_region", "figure"),
    Output("sales_by_category", "figure"),
    Output("top_states", "figure"),
    Output("top_subcats", "figure"),
    Output("lead_time_dist", "figure"),
    Output("top_customers_table", "data"),
    Output("top_customers_table", "columns"),
    Output("top_customers_table", "style_header"),
    Output("top_customers_table", "style_data"),
    Output("top_customers_table", "style_filter"),
    Input("theme", "value"),
    Input("year", "value"),
    Input("quarter", "value"),
    Input("month", "value"),
    Input("region", "value"),
    Input("category", "value"),
)
def update_dashboard(theme, years, quarters, months, regions, categories):
    tokens = theme_tokens(theme)

    years = years or year_options
    quarters = quarters or quarter_options
    months = months or month_options
    regions = regions or region_options
    categories = categories or category_options

    dff = DF[
        (DF["Year"].isin(years))
        & (DF["Quarter"].isin(quarters))
        & (DF["Month"].isin(months))
        & (DF["Region"].isin(regions))
        & (DF["Category"].isin(categories))
    ].copy()

    drilled_region = regions if len(regions) < len(region_options) else []
    drilled_category = categories if len(categories) < len(category_options) else []
    drill_label = " | ".join(
        [
            f"Region: {', '.join(drilled_region)}" if drilled_region else "Region: (all)",
            f"Category: {', '.join(drilled_category)}" if drilled_category else "Category: (all)",
        ]
    )

    # Page + button theme
    page_style = {
        "backgroundColor": tokens["bg"],
        "color": tokens["text"],
        "minHeight": "100vh",
        "padding": "18px 18px 28px 18px",
        "fontFamily": "Arial, sans-serif",
    }
    clear_btn_style = {
        "padding": "10px 12px",
        "borderRadius": "10px",
        "cursor": "pointer",
        "fontWeight": "700",
        "background": tokens["card"],
        "color": tokens["text"],
        "border": tokens["border"],
    }

    # KPIs
    total_sales = float(dff["Sales"].sum())
    avg_sale = float(dff["Sales"].mean()) if len(dff) else 0.0
    order_count = int(dff["Order ID"].nunique())
    customer_count = int(dff["Customer ID"].nunique())
    avg_lead = float(dff["Lead Time (Days)"].mean()) if dff["Lead Time (Days)"].notna().any() else np.nan

    kpis = [
        kpi_card("Total Sales", f"${total_sales:,.2f}", tokens),
        kpi_card("Average Sale", f"${avg_sale:,.2f}", tokens),
        kpi_card("Order Count", f"{order_count:,}", tokens),
        kpi_card("Customer Count", f"{customer_count:,}", tokens),
        kpi_card("Avg Lead Time (Days)", f"{avg_lead:.2f}" if np.isfinite(avg_lead) else "—", tokens),
    ]

    # ----------------------------
    # Sales trend (monthly) + trendline overlay (cyan main line)
    # ----------------------------
    trend = dff.groupby("Year-Month", as_index=False)["Sales"].sum().sort_values("Year-Month")
    trend["Date"] = pd.to_datetime(trend["Year-Month"] + "-01", errors="coerce")

    fig_trend = px.line(trend, x="Date", y="Sales", markers=True, title="Sales Trend (Monthly) + Trend Line")
    fig_trend.update_xaxes(tickformat="%Y-%m")

    # Cyan main line (Sales)
    if fig_trend.data:
        fig_trend.data[0].line.color = CYAN
        fig_trend.data[0].marker.color = CYAN

    # Trendline
    if len(trend) >= 2 and trend["Date"].notna().all():
        x = trend["Date"].map(pd.Timestamp.toordinal).astype(float).to_numpy()
        y = trend["Sales"].astype(float).to_numpy()
        m, b = np.polyfit(x, y, 1)
        y_hat = m * x + b

        trend_color = "rgba(255,255,255,0.65)" if tokens["bg"] != "#FFFFFF" else "rgba(10,31,68,0.55)"
        fig_trend.add_trace(
            go.Scatter(
                x=trend["Date"],
                y=y_hat,
                mode="lines",
                name="Trend",
                line=dict(dash="dash", color=trend_color),
            )
        )

    fig_trend = apply_theme(fig_trend, tokens)

    # ----------------------------
    # Sales by region (pie) - click to drill
    # ----------------------------
    region_sales = dff.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)

    fig_region = px.pie(
        region_sales,
        names="Region",
        values="Sales",
        title="Sales by Region (Click to Drill)",
        color_discrete_sequence=BLUE_SHADES,   # <-- blue shades
    )

    # optional: cleaner slice borders for readability
    fig_region.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.25)")))

    fig_region = apply_theme(fig_region, tokens)

    # ----------------------------
    # Sales by category (bar) - different blue shades + click to drill
    # ----------------------------
    cat_sales = dff.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    fig_cat = px.bar(cat_sales, x="Category", y="Sales", title="Sales by Category (Click to Drill)")
    fig_cat.update_traces(marker_color=BLUE_SHADES[: max(1, len(cat_sales))])
    fig_cat = apply_theme(fig_cat, tokens)

    # ----------------------------
    # Top states (horizontal bar) - different blue shades
    # ----------------------------
    state_sales = dff.groupby("State", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)
    fig_states = px.bar(state_sales, x="Sales", y="State", orientation="h", title="Top 10 States by Sales")
    fig_states.update_traces(marker_color=BLUE_SHADES[: max(1, len(state_sales))])
    fig_states.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_states = apply_theme(fig_states, tokens)

    # ----------------------------
    # Top sub-categories (horizontal bar) - different blue shades
    # ----------------------------
    subcat_sales = (
        dff.groupby("Sub-Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)
    )
    fig_subcats = px.bar(subcat_sales, x="Sales", y="Sub-Category", orientation="h", title="Top 10 Sub-Categories by Sales")
    fig_subcats.update_traces(marker_color=BLUE_SHADES[: max(1, len(subcat_sales))])
    fig_subcats.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_subcats = apply_theme(fig_subcats, tokens)

    # ----------------------------
    # Lead time distribution (blue shades per bin) - binned bar chart
    # ----------------------------
    lead = dff.dropna(subset=["Lead Time (Days)"])
    if len(lead):
        x = lead["Lead Time (Days)"].astype(float).to_numpy()
        counts, edges = np.histogram(x, bins=20)
        centers = (edges[:-1] + edges[1:]) / 2

        colors = [BLUE_SHADES[i % len(BLUE_SHADES)] for i in range(len(counts))]

        fig_lead = go.Figure(
            data=[
                go.Bar(
                    x=centers,
                    y=counts,
                    marker=dict(color=colors),
                    hovertemplate="Lead Time: %{x:.1f} days<br>Count: %{y}<extra></extra>",
                )
            ]
        )
        fig_lead.update_layout(
            title="Lead Time Distribution (Days)",
            xaxis_title="Lead Time (Days)",
            yaxis_title="Count",
        )
    else:
        fig_lead = go.Figure()
        fig_lead.update_layout(
            title="Lead Time Distribution (Days)",
            xaxis_title="Lead Time (Days)",
            yaxis_title="Count",
        )

    fig_lead = apply_theme(fig_lead, tokens)

    # ----------------------------
    # Top customers table
    # ----------------------------
    top_customers = (
        dff.groupby(["Customer Name", "Segment", "Region"], as_index=False)
        .agg(Total_Sales=("Sales", "sum"), Orders=("Order ID", "nunique"))
        .sort_values("Total_Sales", ascending=False)
        .head(50)
    )
    top_customers["Total_Sales"] = top_customers["Total_Sales"].round(2)

    table_data = top_customers.to_dict("records")
    table_cols = [{"name": c.replace("_", " "), "id": c} for c in top_customers.columns]

    header_style = {
        "backgroundColor": tokens["card"],
        "color": tokens["text"],
        "fontWeight": "700",
        "border": tokens["border"],
    }
    data_style = {
        "backgroundColor": tokens["bg"],
        "color": tokens["text"],
        "border": tokens["border"],
    }
    filter_style = {
        "backgroundColor": tokens["bg"],
        "color": tokens["text"],
        "border": tokens["border"],
    }

    return (
        page_style,
        clear_btn_style,
        kpis,
        drill_label,
        fig_trend,
        fig_region,
        fig_cat,
        fig_states,
        fig_subcats,
        fig_lead,
        table_data,
        table_cols,
        header_style,
        data_style,
        filter_style,
    )


if __name__ == "__main__":
    app.run(port=6766, debug=True)