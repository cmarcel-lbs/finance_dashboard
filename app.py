"""
Financial Markets Dashboard — HW3 Interactive Dashboard
Built with Plotly Dash | Data: yfinance + FRED (via pandas_datareader)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── App initialisation ────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Financial Markets Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server   # expose Flask server for Render/Gunicorn

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#0D1117",
    "surface":     "#161B22",
    "surface2":    "#1C2128",
    "border":      "#30363D",
    "accent":      "#58A6FF",
    "accent2":     "#3FB950",
    "accent3":     "#F78166",
    "accent4":     "#D2A8FF",
    "text":        "#E6EDF3",
    "text_muted":  "#8B949E",
    "bull":        "#3FB950",
    "bear":        "#F78166",
    "gold":        "#E3B341",
}

# ── Ticker universe ───────────────────────────────────────────────────────────
TICKERS = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "NVDA":  "NVIDIA",
    "META":  "Meta",
    "JPM":   "JPMorgan",
    "GS":    "Goldman Sachs",
    "XOM":   "ExxonMobil",
    "JNJ":   "Johnson & Johnson",
    "TSLA":  "Tesla",
    "BRK-B": "Berkshire Hathaway",
}

SECTOR_MAP = {
    "AAPL":  "Technology",
    "MSFT":  "Technology",
    "GOOGL": "Technology",
    "AMZN":  "Consumer",
    "NVDA":  "Technology",
    "META":  "Technology",
    "JPM":   "Financials",
    "GS":    "Financials",
    "XOM":   "Energy",
    "JNJ":   "Healthcare",
    "TSLA":  "Consumer",
    "BRK-B": "Financials",
}

YIELD_TICKERS = {
    "DGS3MO": "3-Month",
    "DGS1":   "1-Year",
    "DGS2":   "2-Year",
    "DGS5":   "5-Year",
    "DGS10":  "10-Year",
    "DGS20":  "20-Year",
    "DGS30":  "30-Year",
}

MATURITY_ORDER = ["3-Month", "1-Year", "2-Year", "5-Year", "10-Year", "20-Year", "30-Year"]

# ── Data fetching helpers ─────────────────────────────────────────────────────
def fetch_prices(tickers: list, period: str = "2y") -> pd.DataFrame:
    """Download adjusted closing prices for a list of tickers."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices.dropna(how="all")


def fetch_yield_curve(start: str = "2015-01-01") -> pd.DataFrame:
    """Download US Treasury yields from FRED."""
    end = datetime.today().strftime("%Y-%m-%d")
    frames = {}
    for fred_id, label in YIELD_TICKERS.items():
        try:
            s = pdr.get_data_fred(fred_id, start=start, end=end)
            frames[label] = s.iloc[:, 0]
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.dropna(how="all")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def compute_risk_return(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised return and volatility for each stock."""
    rets = compute_returns(prices)
    ann_ret  = rets.mean() * 252
    ann_vol  = rets.std()  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol
    df = pd.DataFrame({
        "ticker":      ann_ret.index,
        "ann_return":  ann_ret.values * 100,
        "ann_vol":     ann_vol.values * 100,
        "sharpe":      sharpe.values,
        "name":        [TICKERS.get(t, t) for t in ann_ret.index],
        "sector":      [SECTOR_MAP.get(t, "Other") for t in ann_ret.index],
    })
    return df


# ── Layout helpers ────────────────────────────────────────────────────────────
def card(children, style=None):
    base = {
        "background":   COLORS["surface"],
        "border":       f"1px solid {COLORS['border']}",
        "borderRadius": "12px",
        "padding":      "24px",
        "marginBottom": "20px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def section_title(text: str):
    return html.H3(text, style={
        "color":        COLORS["text"],
        "fontFamily":   "'IBM Plex Mono', monospace",
        "fontSize":     "13px",
        "letterSpacing":"2px",
        "textTransform":"uppercase",
        "marginBottom": "16px",
        "borderBottom": f"1px solid {COLORS['border']}",
        "paddingBottom":"10px",
    })


def kpi_tile(label, value, delta=None, delta_positive=True):
    delta_color = COLORS["bull"] if delta_positive else COLORS["bear"]
    return html.Div([
        html.Div(label, style={"color": COLORS["text_muted"], "fontSize": "11px",
                               "letterSpacing": "1.5px", "textTransform": "uppercase",
                               "fontFamily": "'IBM Plex Mono', monospace"}),
        html.Div(value, style={"color": COLORS["text"], "fontSize": "26px",
                               "fontWeight": "700", "margin": "6px 0",
                               "fontFamily": "'IBM Plex Mono', monospace"}),
        html.Div(delta or "", style={"color": delta_color, "fontSize": "12px",
                                     "fontFamily": "'IBM Plex Mono', monospace"})
    ], style={
        "background":   COLORS["surface2"],
        "border":       f"1px solid {COLORS['border']}",
        "borderRadius": "8px",
        "padding":      "18px 22px",
        "flex":         "1",
        "minWidth":     "160px",
    })


# ── App layout ────────────────────────────────────────────────────────────────
app.layout = html.Div(style={
    "background":   COLORS["bg"],
    "minHeight":    "100vh",
    "fontFamily":   "'Inter', 'Segoe UI', sans-serif",
    "color":        COLORS["text"],
    "padding":      "0",
}, children=[

    # Google Fonts
    html.Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap"),

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("▲", style={"color": COLORS["accent"], "fontSize": "22px", "marginRight": "12px"}),
            html.Div([
                html.H1("Financial Markets Dashboard",
                        style={"margin": "0", "fontSize": "20px", "fontWeight": "600",
                               "color": COLORS["text"], "letterSpacing": "0.5px"}),
                html.Div("Equity Prices · Risk-Return · Yield Curve",
                         style={"color": COLORS["text_muted"], "fontSize": "12px",
                                "fontFamily": "'IBM Plex Mono', monospace", "marginTop": "2px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div(id="last-updated", style={
            "color": COLORS["text_muted"], "fontSize": "11px",
            "fontFamily": "'IBM Plex Mono', monospace",
        }),
    ], style={
        "background":    COLORS["surface"],
        "borderBottom":  f"1px solid {COLORS['border']}",
        "padding":       "18px 32px",
        "display":       "flex",
        "justifyContent":"space-between",
        "alignItems":    "center",
        "position":      "sticky",
        "top":           "0",
        "zIndex":        "100",
    }),

    # ── Main content ──────────────────────────────────────────────────────────
    html.Div(style={"maxWidth": "1400px", "margin": "0 auto", "padding": "28px 32px"}, children=[

        # ── Controls row ──────────────────────────────────────────────────────
        card([
            html.Div([
                # Ticker picker
                html.Div([
                    html.Label("Select Stocks", style={"color": COLORS["text_muted"], "fontSize": "11px",
                                                       "letterSpacing": "1.5px", "textTransform": "uppercase",
                                                       "fontFamily": "'IBM Plex Mono', monospace", "marginBottom": "8px",
                                                       "display": "block"}),
                    dcc.Dropdown(
                        id="ticker-dropdown",
                        options=[{"label": f"{t} — {n}", "value": t} for t, n in TICKERS.items()],
                        value=["AAPL", "MSFT", "NVDA", "GOOGL", "JPM"],
                        multi=True,
                        style={"background": COLORS["surface2"]},
                        className="dark-dropdown",
                    ),
                ], style={"flex": "3", "marginRight": "24px"}),

                # Period picker
                html.Div([
                    html.Label("Time Period", style={"color": COLORS["text_muted"], "fontSize": "11px",
                                                     "letterSpacing": "1.5px", "textTransform": "uppercase",
                                                     "fontFamily": "'IBM Plex Mono', monospace", "marginBottom": "8px",
                                                     "display": "block"}),
                    dcc.RadioItems(
                        id="period-radio",
                        options=[
                            {"label": "6M", "value": "6mo"},
                            {"label": "1Y", "value": "1y"},
                            {"label": "2Y", "value": "2y"},
                            {"label": "5Y", "value": "5y"},
                        ],
                        value="2y",
                        inline=True,
                        style={"color": COLORS["text"]},
                        inputStyle={"marginRight": "5px", "accentColor": COLORS["accent"]},
                        labelStyle={"marginRight": "18px", "fontSize": "13px",
                                    "fontFamily": "'IBM Plex Mono', monospace"},
                    ),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "alignItems": "flex-end"}),
        ]),

        # ── KPI row ───────────────────────────────────────────────────────────
        html.Div(id="kpi-row", style={"display": "flex", "gap": "16px",
                                       "marginBottom": "20px", "flexWrap": "wrap"}),

        # ── Price chart ───────────────────────────────────────────────────────
        card([
            section_title("📈  Normalised Price Performance (Base = 100)"),
            dcc.Graph(id="price-chart", config={"displayModeBar": False},
                      style={"height": "380px"}),
        ]),

        # ── Risk-Return scatter ───────────────────────────────────────────────
        html.Div([
            html.Div([
                card([
                    section_title("⚡  Risk–Return Scatter"),
                    dcc.Graph(id="risk-return-chart", config={"displayModeBar": False},
                              style={"height": "400px"}),
                ], style={"height": "100%", "marginBottom": "0"}),
            ], style={"flex": "1", "marginRight": "16px"}),

            html.Div([
                card([
                    section_title("📊  Daily Returns Distribution"),
                    dcc.Graph(id="returns-dist-chart", config={"displayModeBar": False},
                              style={"height": "400px"}),
                ], style={"height": "100%", "marginBottom": "0"}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "20px"}),

        # ── Correlation heatmap ───────────────────────────────────────────────
        html.Div([
            html.Div([
                card([
                    section_title("🔗  Returns Correlation Matrix"),
                    dcc.Graph(id="corr-heatmap", config={"displayModeBar": False},
                              style={"height": "400px"}),
                ], style={"height": "100%", "marginBottom": "0"}),
            ], style={"flex": "1", "marginRight": "16px"}),

            html.Div([
                card([
                    section_title("🗓️  Monthly Returns Heatmap"),
                    dcc.Dropdown(
                        id="heatmap-ticker-dropdown",
                        options=[{"label": f"{t} — {n}", "value": t} for t, n in TICKERS.items()],
                        value="AAPL",
                        multi=False,
                        style={"background": COLORS["surface2"], "marginBottom": "12px"},
                    ),
                    dcc.Graph(id="monthly-heatmap", config={"displayModeBar": False},
                              style={"height": "350px"}),
                ], style={"height": "100%", "marginBottom": "0"}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "20px"}),

        # ── Yield curve section ────────────────────────────────────────────────
        card([
            section_title("🏛️  US Treasury Yield Curve"),
            html.Div([
                html.Div([
                    html.Label("Select date for snapshot",
                               style={"color": COLORS["text_muted"], "fontSize": "11px",
                                      "letterSpacing": "1.5px", "textTransform": "uppercase",
                                      "fontFamily": "'IBM Plex Mono', monospace",
                                      "marginBottom": "8px", "display": "block"}),
                    dcc.DatePickerSingle(
                        id="yield-date-picker",
                        date=(datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d"),
                        display_format="YYYY-MM-DD",
                        style={"marginBottom": "16px"},
                    ),
                ], style={"marginRight": "32px"}),
                html.Div([
                    html.Label("Compare to",
                               style={"color": COLORS["text_muted"], "fontSize": "11px",
                                      "letterSpacing": "1.5px", "textTransform": "uppercase",
                                      "fontFamily": "'IBM Plex Mono', monospace",
                                      "marginBottom": "8px", "display": "block"}),
                    dcc.Checklist(
                        id="yield-compare-checklist",
                        options=[
                            {"label": "  1 year ago", "value": "1y"},
                            {"label": "  2 years ago", "value": "2y"},
                            {"label": "  5 years ago", "value": "5y"},
                        ],
                        value=["1y"],
                        inline=True,
                        inputStyle={"marginRight": "5px", "accentColor": COLORS["accent"]},
                        labelStyle={"marginRight": "20px", "fontSize": "13px",
                                    "color": COLORS["text"],
                                    "fontFamily": "'IBM Plex Mono', monospace"},
                    ),
                ]),
            ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "8px"}),

            html.Div([
                html.Div([
                    dcc.Graph(id="yield-curve-snapshot", config={"displayModeBar": False},
                              style={"height": "360px"}),
                ], style={"flex": "1", "marginRight": "16px"}),
                html.Div([
                    dcc.Graph(id="spread-chart", config={"displayModeBar": False},
                              style={"height": "360px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),

        # ── Footer ────────────────────────────────────────────────────────────
        html.Div([
            html.Span("Data sourced from ", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
            html.Span("Yahoo Finance", style={"color": COLORS["accent"], "fontSize": "11px"}),
            html.Span(" & ", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
            html.Span("FRED (Federal Reserve)", style={"color": COLORS["accent"], "fontSize": "11px"}),
            html.Span(" · HW3 Financial Dashboard · CAM", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
        ], style={"textAlign": "center", "padding": "24px 0 8px",
                  "fontFamily": "'IBM Plex Mono', monospace"}),
    ]),
])


# ── CALLBACKS ─────────────────────────────────────────────────────────────────

def plotly_layout(title="", height=400):
    return dict(
        title=dict(text=title, font=dict(size=13, color=COLORS["text_muted"])),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
        xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"],
                   zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"],
                   zerolinecolor=COLORS["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"],
                    font=dict(size=11)),
        margin=dict(l=50, r=20, t=40, b=50),
        height=height,
    )


@callback(
    Output("last-updated", "children"),
    Output("kpi-row", "children"),
    Output("price-chart", "figure"),
    Output("risk-return-chart", "figure"),
    Output("returns-dist-chart", "figure"),
    Output("corr-heatmap", "figure"),
    Input("ticker-dropdown", "value"),
    Input("period-radio", "value"),
)
def update_stock_charts(tickers, period):
    if not tickers:
        empty = go.Figure()
        empty.update_layout(**plotly_layout())
        return "No data", [], empty, empty, empty, empty

    # ── Fetch data ────────────────────────────────────────────────────────────
    prices = fetch_prices(tickers, period=period)
    prices = prices[[t for t in tickers if t in prices.columns]]
    returns = compute_returns(prices)
    rr = compute_risk_return(prices)

    # ── KPI tiles ─────────────────────────────────────────────────────────────
    kpis = []
    for _, row in rr.iterrows():
        pos = row["ann_return"] >= 0
        kpis.append(kpi_tile(
            row["ticker"],
            f"{row['ann_return']:+.1f}%",
            f"Vol {row['ann_vol']:.1f}%  |  Sharpe {row['sharpe']:.2f}",
            delta_positive=pos,
        ))

    # ── Normalised price chart ─────────────────────────────────────────────────
    fig_price = go.Figure()
    colour_seq = [COLORS["accent"], COLORS["accent2"], COLORS["accent3"],
                  COLORS["accent4"], COLORS["gold"], "#79C0FF", "#56D364",
                  "#FF7B72", "#BC8CFF", "#F0883E", "#58A6FF", "#3FB950"]
    for i, t in enumerate(prices.columns):
        norm = prices[t] / prices[t].dropna().iloc[0] * 100
        fig_price.add_trace(go.Scatter(
            x=norm.index, y=norm.values, name=t, mode="lines",
            line=dict(color=colour_seq[i % len(colour_seq)], width=2),
            hovertemplate=f"<b>{t}</b><br>%{{x|%b %d %Y}}<br>Index: %{{y:.1f}}<extra></extra>",
        ))
    fig_price.add_hline(y=100, line_dash="dot", line_color=COLORS["border"],
                        annotation_text="Base (100)", annotation_font_color=COLORS["text_muted"])
    fig_price.update_layout(**plotly_layout(height=380))
    fig_price.update_xaxes(showgrid=False)

    # ── Risk-return scatter ────────────────────────────────────────────────────
    sector_colours = {
        "Technology": COLORS["accent"],
        "Financials":  COLORS["accent2"],
        "Consumer":    COLORS["accent3"],
        "Healthcare":  COLORS["accent4"],
        "Energy":      COLORS["gold"],
        "Other":       COLORS["text_muted"],
    }
    fig_rr = go.Figure()
    for sector, grp in rr.groupby("sector"):
        fig_rr.add_trace(go.Scatter(
            x=grp["ann_vol"], y=grp["ann_return"],
            mode="markers+text",
            name=sector,
            text=grp["ticker"],
            textposition="top center",
            textfont=dict(size=10, color=COLORS["text"]),
            marker=dict(
                size=grp["sharpe"].clip(0.1, 3) * 14,
                color=sector_colours.get(sector, COLORS["text_muted"]),
                opacity=0.85,
                line=dict(width=1, color=COLORS["bg"]),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Ann. Return: %{y:.1f}%<br>"
                "Ann. Vol: %{x:.1f}%<br>"
                "<extra></extra>"
            ),
        ))
    fig_rr.add_hline(y=0, line_dash="dot", line_color=COLORS["border"])
    fig_rr.update_layout(
        **plotly_layout(height=400),
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
    )

    # ── Returns distribution ───────────────────────────────────────────────────
    fig_dist = go.Figure()
    for i, t in enumerate(returns.columns):
        fig_dist.add_trace(go.Violin(
            y=returns[t].dropna() * 100,
            name=t,
            box_visible=True,
            meanline_visible=True,
            fillcolor=colour_seq[i % len(colour_seq)],
            line_color=colour_seq[i % len(colour_seq)],
            opacity=0.7,
        ))
    fig_dist.update_layout(
        **plotly_layout(height=400),
        yaxis_title="Daily Return (%)",
        showlegend=False,
        violinmode="overlay",
    )

    # ── Correlation heatmap ────────────────────────────────────────────────────
    corr = returns.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, COLORS["bear"]], [0.5, COLORS["surface2"]], [1, COLORS["bull"]]],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="%{x} / %{y}<br>Corr: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=COLORS["text"])),
    ))
    fig_corr.update_layout(**plotly_layout(height=400))

    ts = datetime.now().strftime("Updated %H:%M · %b %d %Y")
    return ts, kpis, fig_price, fig_rr, fig_dist, fig_corr


@callback(
    Output("monthly-heatmap", "figure"),
    Input("heatmap-ticker-dropdown", "value"),
    Input("period-radio", "value"),
)
def update_monthly_heatmap(ticker, period):
    if not ticker:
        return go.Figure()
    prices = fetch_prices([ticker], period=period)
    if ticker not in prices.columns:
        return go.Figure()

    monthly = (prices[ticker]
               .resample("ME").last()
               .pct_change()
               .dropna() * 100)
    monthly_df = monthly.to_frame("ret")
    monthly_df["year"]  = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.strftime("%b")
    pivot = monthly_df.pivot_table(index="year", columns="month", values="ret")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.astype(str).tolist(),
        colorscale=[[0, COLORS["bear"]], [0.5, "#161B22"], [1, COLORS["bull"]]],
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont=dict(size=10),
        colorbar=dict(tickfont=dict(color=COLORS["text"]),
                      ticksuffix="%"),
        hovertemplate="%{x} %{y}<br>Return: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(title=f"{ticker} Monthly Returns", height=350))
    return fig


@callback(
    Output("yield-curve-snapshot", "figure"),
    Output("spread-chart", "figure"),
    Input("yield-date-picker", "date"),
    Input("yield-compare-checklist", "value"),
)
def update_yield_charts(selected_date, compare_periods):
    yc = fetch_yield_curve(start="2015-01-01")
    if yc.empty:
        empty = go.Figure()
        empty.update_layout(**plotly_layout())
        return empty, empty

    cols = [c for c in MATURITY_ORDER if c in yc.columns]
    yc = yc[cols]

    # ── Yield curve snapshot ──────────────────────────────────────────────────
    fig_yc = go.Figure()
    snap_colors = [COLORS["accent"], COLORS["gold"], COLORS["accent3"], COLORS["accent2"]]
    labels = ["Today"] + [f"{p} ago" for p in (compare_periods or [])]
    dates_to_plot = [selected_date] + [
        (pd.Timestamp(selected_date) - pd.DateOffset(years=int(p[0]))).strftime("%Y-%m-%d")
        for p in (compare_periods or [])
    ]

    for i, (date_str, label) in enumerate(zip(dates_to_plot, labels)):
        try:
            row = yc.loc[:date_str].iloc[-1]
        except Exception:
            continue
        fig_yc.add_trace(go.Scatter(
            x=cols, y=row.values,
            mode="lines+markers",
            name=f"{label} ({row.name.strftime('%Y-%m-%d')})",
            line=dict(color=snap_colors[i % len(snap_colors)], width=2.5),
            marker=dict(size=7),
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))

    fig_yc.update_layout(
        **plotly_layout(height=360),
        xaxis_title="Maturity",
        yaxis_title="Yield (%)",
        yaxis_ticksuffix="%",
    )

    # ── 10Y-2Y spread (inversion indicator) ───────────────────────────────────
    fig_spread = go.Figure()
    if "10-Year" in yc.columns and "2-Year" in yc.columns:
        spread = (yc["10-Year"] - yc["2-Year"]).dropna()
        colours_spread = [COLORS["bull"] if v >= 0 else COLORS["bear"] for v in spread]
        fig_spread.add_trace(go.Bar(
            x=spread.index, y=spread.values,
            marker_color=colours_spread,
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra></extra>",
            name="10Y–2Y Spread",
        ))
        fig_spread.add_hline(y=0, line_dash="dash", line_color=COLORS["text_muted"])
        fig_spread.add_annotation(
            x=0.02, y=0.92, xref="paper", yref="paper",
            text="🔴 Inverted = recession signal",
            showarrow=False,
            font=dict(color=COLORS["bear"], size=11),
            bgcolor=COLORS["surface2"],
        )
    fig_spread.update_layout(
        **plotly_layout(title="10Y – 2Y Yield Spread (Inversion Indicator)", height=360),
        yaxis_title="Spread (%)",
        yaxis_ticksuffix="%",
        showlegend=False,
        bargap=0.0,
    )
    fig_spread.update_xaxes(showgrid=False)

    return fig_yc, fig_spread


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
