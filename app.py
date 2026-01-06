# app.py
import functools

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import pandas as pd

from data_prep import load_brfss, build_layerQ
from aggregations import (
    aggregate_overall,
    aggregate_by_year,
    aggregate_by_state,
    aggregate_by_breakout_category,
)
from geographic_grouping import (
    aggregate_by_region,
    group_age_ranges,
    group_income_ranges,
    group_education_levels,
)

# ---------- Load data at startup ----------
print("=" * 60)
print("🚀 Starting BRFSS Dashboard...")
print("=" * 60)

df = load_brfss()
layerQ = build_layerQ(df)

years = sorted(int(y) for y in df["Year"].unique())
min_year, max_year = years[0], years[-1]
all_states = sorted(df["Locationabbr"].unique())

print(f"✓ Data loaded: {min_year}-{max_year}, {len(all_states)} states/territories")
print("=" * 60)

# ---------- Color Schemes ----------
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',   # use for “yes / positive”
    'warning': '#F18F01',
    'danger': '#C73E1D',    # use for “no / negative”
    'info': '#4EA5D9',
    'background': '#F8F9FA',
    'text': '#212529',
}

# Base palette for "other" responses
RESPONSE_PALETTE = px.colors.qualitative.Set2

# Semantic mapping for key responses (remember: Response is lower-cased in data_prep)
RESPONSE_COLOR_BASE = {
    "yes": COLORS["success"],   # green
    "no": COLORS["danger"],     # red
}

def get_response_color_map(series):
    """
    Build a stable color map for the given Response series:
    - 'yes' -> green
    - 'no'  -> red
    - all other categories cycle through a qualitative palette
    """
    unique_responses = sorted({str(v) for v in series.dropna().unique()})
    color_map = {}
    palette_idx = 0

    for resp in unique_responses:
        key = resp.lower()
        if key in RESPONSE_COLOR_BASE:
            color_map[resp] = RESPONSE_COLOR_BASE[key]
        else:
            color_map[resp] = RESPONSE_PALETTE[palette_idx % len(RESPONSE_PALETTE)]
            palette_idx += 1

    return color_map

# ---------- Helper Functions ----------

class_options = [{"label": c, "value": c} for c in sorted(layerQ["Class"].unique())]

compare_question_options = [
    {
        "label": f"{row['Class']} — {row['Topic']} — {row['Question']}",
        "value": row["Question"],
    }
    for _, row in layerQ.sort_values(["Class", "Topic", "Question"]).iterrows()
]


def get_topic_options(selected_class: str):
    if selected_class is None:
        return []
    subset = layerQ[layerQ["Class"] == selected_class]
    return [{"label": t, "value": t} for t in sorted(subset["Topic"].unique())]


def get_question_options(selected_class: str, selected_topic: str):
    if selected_class is None or selected_topic is None:
        return []
    subset = layerQ[(layerQ["Class"] == selected_class) & (layerQ["Topic"] == selected_topic)]
    return [{"label": row["Question"], "value": row["Question"]} for _, row in subset.iterrows()]


# ---------- Figure Creation Functions ----------

def create_base_layout(title, x_label="", y_label="Percent (%)", height=None):
    """Create consistent base layout for all plots (no per-figure legend)."""
    layout = dict(
        title=dict(
            text=title,
            font=dict(size=15, color=COLORS['text'], family="Arial, sans-serif", weight=600),
            x=0.5,
            xanchor='center',
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=12, color=COLORS['text'])),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=12, color=COLORS['text'])),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
        ),
        plot_bgcolor='rgba(248,249,250,0.3)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11, color=COLORS['text']),
        # ❗ Hide legends inside each figure – we’ll show one global legend instead
        showlegend=False,
        margin=dict(l=60, r=30, t=100, b=60),
        hovermode='closest',
    )

    if height:
        layout['height'] = height

    return layout


def make_bar_figure(df_plot, x, y, color, title, x_label=None, y_label="Percent (%)", 
                    error_col=None, show_errorbars=True, rotate_x=False, height=None):
    """Enhanced bar chart with consistent semantic colors."""
    if df_plot is None or df_plot.empty:
        fig = go.Figure()
        layout = create_base_layout(f"{title} (No data)", x_label or "", y_label, height)
        layout['annotations'] = [dict(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )]
        fig.update_layout(**layout)
        return fig

    # Build a color map based on the responses actually present
    color_map = get_response_color_map(df_plot[color])

    if show_errorbars and error_col and error_col in df_plot.columns:
        fig = px.bar(
            df_plot,
            x=x,
            y=y,
            color=color,
            error_y=error_col,
            color_discrete_map=color_map,
            barmode="group",
        )
    else:
        fig = px.bar(
            df_plot,
            x=x,
            y=y,
            color=color,
            color_discrete_map=color_map,
            barmode="group",
        )

    layout = create_base_layout(title, x_label or "", y_label, height)

    if rotate_x:
        layout["xaxis"]["tickangle"] = -45
        layout["margin"]["b"] = 100

    fig.update_layout(**layout)
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
    )

    return fig


def make_line_figure(df_plot, x, y, color, title, x_label="", y_label="Percent (%)", height=None):
    """Line chart for temporal trends with consistent semantic colors."""
    if df_plot is None or df_plot.empty:
        fig = go.Figure()
        layout = create_base_layout(f"{title} (No data)", x_label, y_label, height)
        fig.update_layout(**layout)
        return fig

    color_map = get_response_color_map(df_plot[color])

    fig = px.line(
        df_plot,
        x=x,
        y=y,
        color=color,
        markers=True,
        color_discrete_map=color_map,
    )

    layout = create_base_layout(title, x_label, y_label, height)
    fig.update_layout(**layout)
    fig.update_traces(
        mode="lines+markers",
        line=dict(width=2.5),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    )

    return fig



def make_choropleth_figure(state_df, title, target_response: str = "yes", height=None):
    """Enhanced choropleth map with detailed hover information."""
    if state_df is None or state_df.empty:
        fig = go.Figure()
        layout = create_base_layout(f"{title} (No data)", height=height)
        fig.update_layout(**layout)
        fig.update_geos(scope="usa")
        return fig

    df_plot = state_df.copy()

    if target_response in df_plot["Response"].unique():
        df_plot = df_plot[df_plot["Response"] == target_response]
        response_label = target_response.title()
    else:
        df_plot = df_plot.sort_values("agg_percent", ascending=False).groupby("Locationabbr", as_index=False).first()
        response_label = "Top Category"

    # Create custom hover text
    df_plot['hover_text'] = df_plot.apply(
        lambda row: (
            f"<b>{row['Locationabbr']}</b><br>"
            f"Response: {row['Response'].title()}<br>"
            f"Percent: {row['agg_percent']:.2f}%<br>"
            f"95% CI: [{row['agg_low_ci_limit']:.2f}%, {row['agg_high_ci_limit']:.2f}%]<br>"
            f"Sample Size: {int(row['agg_ss']):,}"
        ),
        axis=1
    )

    fig = go.Figure(data=go.Choropleth(
        locations=df_plot['Locationabbr'],
        z=df_plot['agg_percent'],
        locationmode='USA-states',
        colorscale=[[0, '#f7fbff'], [0.5, '#4292c6'], [1, '#084594']],
        text=df_plot['hover_text'],
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title="Percent (%)",
            thickness=15,
            len=0.6,
            x=1.0
        ),
        marker_line_color='white',
        marker_line_width=1.5
    ))

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Response: {response_label}</sub>",
            font=dict(size=15),
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='white',
        geo=dict(
            scope='usa',
            bgcolor='rgba(248,249,250,0.3)',
            lakecolor='white',
            landcolor='rgba(242,242,242,0.5)'
        ),
        height=height
    )
    
    return fig


def generate_insight_summary(overall_df, temporal_df, state_df, gender_df, age_df, race_df, edu_df, income_df):
    """Generate insights summary."""
    lines = []

    if overall_df is not None and not overall_df.empty:
        total_n = overall_df['agg_ss'].iloc[0]
        lines.append(f"📊 **Sample Size:** {total_n:,.0f} responses")
        top_row = overall_df.loc[overall_df["agg_percent"].idxmax()]
        lines.append(f"🎯 **National Prevalence:** {top_row['agg_percent']:.1f}% (*{top_row['Response'].title()}*)")

    if temporal_df is not None and not temporal_df.empty:
        temp = temporal_df.groupby("Year")["agg_percent"].mean()
        if len(temp) >= 2:
            first, last = temp.iloc[0], temp.iloc[-1]
            diff = last - first
            if abs(diff) > 1.0:
                arrow = "📈" if diff > 0 else "📉"
                direction = "increase" if diff > 0 else "decrease"
                lines.append(f"{arrow} **Trend:** {abs(diff):.1f} point {direction} ({temp.index[0]}→{temp.index[-1]})")
            else:
                lines.append("➡️ **Trend:** Stable over time")

    if state_df is not None and not state_df.empty:
        st = state_df.groupby("Locationabbr")["agg_percent"].mean().sort_values(ascending=False)
        high_state, high_val = st.index[0], st.iloc[0]
        low_state, low_val = st.index[-1], st.iloc[-1]
        lines.append(f"🗺️ **Geographic Range:** {high_state} ({high_val:.1f}%) ↔ {low_state} ({low_val:.1f}%)")

    def get_gap(df_demo, label, emoji):
        if df_demo is None or df_demo.empty:
            return None, None, None
        means = df_demo.groupby("Break_Out")["agg_percent"].mean()
        if len(means) < 2:
            return None, None, None
        return means.max() - means.min(), label, emoji

    gaps = []
    for demo_df, name, emoji in [
        (gender_df, "gender", "👥"), (age_df, "age", "📅"),
        (race_df, "race/ethnicity", "🌍"), (edu_df, "education", "🎓"),
        (income_df, "income", "💰"),
    ]:
        spread, label, em = get_gap(demo_df, name, emoji)
        if spread is not None:
            gaps.append((spread, label, em))

    if gaps:
        biggest_gap, group_name, emoji = max(gaps, key=lambda x: x[0])
        lines.append(f"{emoji} **Key Disparity:** {biggest_gap:.1f} points across {group_name}")

    return "\n\n".join(lines) if lines else "Select a question to see insights."


def build_ci_table(selected_breakdown, overall_df, gender_df, age_df, race_df, edu_df, income_df):
    """Build confidence intervals table."""
    
    if selected_breakdown == "overall":
        if overall_df is None or overall_df.empty:
            return pd.DataFrame()
        df = overall_df[['Response', 'agg_percent', 'agg_low_ci_limit', 'agg_high_ci_limit', 'agg_ss']].copy()
        df.columns = ['Response', 'Percent (%)', '95% CI Lower', '95% CI Upper', 'Sample Size']
        
    elif selected_breakdown == "gender":
        if gender_df is None or gender_df.empty:
            return pd.DataFrame()
        df = gender_df[['Break_Out', 'Response', 'agg_percent', 'agg_low_ci_limit', 'agg_high_ci_limit', 'agg_ss']].copy()
        df.columns = ['Gender', 'Response', 'Percent (%)', '95% CI Lower', '95% CI Upper', 'Sample Size']
        
    elif selected_breakdown == "age":
        if age_df is None or age_df.empty:
            return pd.DataFrame()
        df = age_df[['Break_Out', 'Response', 'agg_percent', 'agg_low_ci_limit', 'agg_high_ci_limit', 'agg_ss']].copy()
        df.columns = ['Age Group', 'Response', 'Percent (%)', '95% CI Lower', '95% CI Upper', 'Sample Size']
        
    elif selected_breakdown == "race":
        if race_df is None or race_df.empty:
            return pd.DataFrame()
        df = race_df[['Break_Out', 'Response', 'agg_percent', 'agg_low_ci_limit', 'agg_high_ci_limit', 'agg_ss']].copy()
        df.columns = ['Race/Ethnicity', 'Response', 'Percent (%)', '95% CI Lower', '95% CI Upper', 'Sample Size']
        
    elif selected_breakdown == "education":
        if edu_df is None or edu_df.empty:
            return pd.DataFrame()
        df = edu_df[['Break_Out', 'Response', 'agg_percent', 'agg_low_ci_limit', 'agg_high_ci_limit', 'agg_ss']].copy()
        df.columns = ['Education', 'Response', 'Percent (%)', '95% CI Lower', '95% CI Upper', 'Sample Size']
        
    elif selected_breakdown == "income":
        if income_df is None or income_df.empty:
            return pd.DataFrame()
        df = income_df[['Break_Out', 'Response', 'agg_percent', 'agg_low_ci_limit', 'agg_high_ci_limit', 'agg_ss']].copy()
        df.columns = ['Income', 'Response', 'Percent (%)', '95% CI Lower', '95% CI Upper', 'Sample Size']
    else:
        return pd.DataFrame()
    
    # Round numeric columns
    for col in ['Percent (%)', '95% CI Lower', '95% CI Upper']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    if 'Sample Size' in df.columns:
        df['Sample Size'] = df['Sample Size'].astype(int)
    
    return df


def extract_main_series(temporal_df, question_label, target_response="__auto__"):
    """Extract one time series from a temporal aggregation for comparing questions.

    - If target_response == "__auto__" or None:
        Prefer 'yes' if present, then '1', otherwise take the dominant
        response per year (highest agg_percent).
    - If target_response is a specific label:
        Use that label (case-insensitive). If it's missing, return None.
    """
    if temporal_df is None or temporal_df.empty:
        return None

    df_temp = temporal_df.copy()
    df_temp["Response_norm"] = df_temp["Response"].astype(str).str.lower()

    # AUTO MODE
    if not target_response or target_response == "__auto__":
        # 1) Prefer 'yes'
        if (df_temp["Response_norm"] == "yes").any():
            df_temp = df_temp[df_temp["Response_norm"] == "yes"]
        # 2) Or '1' (e.g., coded yes)
        elif (df_temp["Response_norm"] == "1").any():
            df_temp = df_temp[df_temp["Response_norm"] == "1"]
        # 3) Otherwise, pick the dominant response per year
        else:
            df_temp = (
                df_temp
                .sort_values(["Year", "agg_percent"], ascending=[True, False])
                .groupby("Year", as_index=False)
                .head(1)
            )
    else:
        # EXPLICIT RESPONSE
        target_norm = str(target_response).lower()
        mask = df_temp["Response_norm"] == target_norm
        if not mask.any():
            return None
        df_temp = df_temp[mask]

    if df_temp.empty:
        return None

    out = df_temp[["Year", "agg_percent"]].copy()
    label = question_label or ""
    out["QuestionLabel"] = label[:60] + "..." if len(label) > 60 else label
    return out


# ---------- Cached Aggregation ----------

@functools.lru_cache(maxsize=512)
def compute_panels_cached(question_text: str, year_min: int, year_max: int, states_key: tuple, clean_flag: bool):
    """Compute all panels for a given question and filters."""
    if not question_text or question_text not in df['Question'].values:
        raise ValueError(f"Invalid question: {question_text}")
    
    df_q = df[df["Question"] == question_text].copy()
    df_q = df_q[(df_q["Year"] >= year_min) & (df_q["Year"] <= year_max)]
    if states_key:
        df_q = df_q[df_q["Locationabbr"].isin(states_key)]

    overall = aggregate_overall(df_q, clean=clean_flag)
    temporal = aggregate_by_year(df_q, clean=clean_flag)
    state = aggregate_by_state(df_q, clean=clean_flag)
    gender = aggregate_by_breakout_category(df_q, "CAT2", clean=clean_flag)
    age = aggregate_by_breakout_category(df_q, "CAT3", clean=clean_flag)
    race = aggregate_by_breakout_category(df_q, "CAT4", clean=clean_flag)
    edu = aggregate_by_breakout_category(df_q, "CAT5", clean=clean_flag)
    income = aggregate_by_breakout_category(df_q, "CAT6", clean=clean_flag)

    return overall, temporal, state, gender, age, race, edu, income


# ---------- Dash App ----------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "BRFSS Dashboard"

# ---------- SIDEBAR ----------

sidebar_style = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '340px',
    'padding': '20px',
    'background': 'linear-gradient(180deg, #2c3e50 0%, #34495e 100%)',
    'color': '#ecf0f1',
    'overflowY': 'auto',
    'zIndex': '1000',
    'boxShadow': '4px 0 12px rgba(0,0,0,0.15)'
}

content_style = {
    'marginLeft': '360px',
    'padding': '20px',
    'backgroundColor': '#f8f9fa',
    'minHeight': '100vh'
}

label_style = {
    'color': '#ecf0f1',
    'fontSize': '0.8rem',
    'fontWeight': '600',
    'marginBottom': '5px',
    'display': 'block',
    'textTransform': 'uppercase',
    'letterSpacing': '0.5px'
}

dropdown_style = {
    'fontSize': '0.85rem',
    'width': '100%'
}

sidebar = html.Div(
    style=sidebar_style,
    children=[
        html.Div([
            html.H4("BRFSS", style={'color': '#ecf0f1', 'marginBottom': '0', 'fontWeight': '700'}),
            html.P("Health Insights Dashboard", style={'color': '#bdc3c7', 'fontSize': '0.85rem', 'marginTop': '-5px'}),
        ]),
        
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)', 'margin': '15px 0'}),
        
        html.Div([
            html.Label("Health Class", style=label_style),
            dcc.Dropdown(
                id="class-dropdown",
                options=class_options,
                value=class_options[0]["value"] if class_options else None,
                style=dropdown_style,
                clearable=False
            ),
        ], style={'marginBottom': '15px'}),
        
        html.Div([
            html.Label("Topic", style=label_style),
            dcc.Dropdown(
                id="topic-dropdown",
                options=[],
                value=None,
                style=dropdown_style,
                clearable=False
            ),
        ], style={'marginBottom': '15px'}),
        
        html.Div([
            html.Label("Question", style=label_style),
            dcc.Dropdown(
                id="question-dropdown",
                options=[],
                value=None,
                style=dropdown_style,
                clearable=False,
                optionHeight=60  # More height per option for wrapping
            ),
        ], style={'marginBottom': '15px'}),
        
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)', 'margin': '15px 0'}),
        
        html.Div([
            html.Label("Year Range", style=label_style),
            dcc.RangeSlider(
                id="year-range",
                min=min_year,
                max=max_year,
                value=[min_year, max_year],
                step=1,
                marks={int(y): {'label': str(int(y)), 'style': {'color': '#ecf0f1', 'fontSize': '0.7rem'}} 
                       for y in years[::3]},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("States/Territories", style=label_style),
            dcc.Dropdown(
                id="state-select",
                options=[{"label": s, "value": s} for s in all_states],
                value=[],
                multi=True,
                placeholder="All states",
                style=dropdown_style
            ),
        ], style={'marginBottom': '15px'}),
        
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)', 'margin': '15px 0'}),
        
        html.Div([
           html.Label("Options", style=label_style),
           dcc.Checklist(
               id="options-toggle",
               options=[
                   {"label": " Show CI", "value": "errorbars"},
               ],
               value=["errorbars"],  # default: show confidence intervals
               style={'color': '#ecf0f1', 'fontSize': '0.85rem'},
               inputStyle={'marginRight': '8px'}
           ),
        ], style={'marginBottom': '15px'}),
        
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)', 'margin': '15px 0'}),
        
        html.Div(id="cache-stats", style={
            'color': '#95a5a6',
            'fontSize': '0.75rem',
            'textAlign': 'center',
            'padding': '8px',
            'backgroundColor': 'rgba(0,0,0,0.2)',
            'borderRadius': '4px'
        }),
    ]
)

# ---------- MAIN CONTENT ----------

main_content = html.Div(
    style=content_style,
    children=[
        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-overview",
            children=[
                dbc.Tab(label="Overview", tab_id="tab-overview"),
                dbc.Tab(label="Demographics", tab_id="tab-demographics"),
                dbc.Tab(label="Geography", tab_id="tab-geography"),
                dbc.Tab(label="Compare", tab_id="tab-compare"),
                dbc.Tab(label="Insights & Data", tab_id="tab-insights"),
            ],
        ),

        # 🔍 Global legend (one place, under tabs)
        html.Div(id="global-legend", style={'marginTop': '10px', 'marginBottom': '10px'}),

        html.Div(id="tab-content", style={'paddingTop': '10px'})
    ]
)


# ---------- App Layout ----------

app.layout = html.Div([sidebar, main_content])

# ---------- Callbacks ----------

@app.callback(
    Output("cache-stats", "children"),
    Input("question-dropdown", "value")
)
def show_cache_stats(_):
    info = compute_panels_cached.cache_info()
    if info.hits + info.misses == 0:
        return ""
    hit_rate = info.hits / (info.hits + info.misses) * 100
    return f"⚡ Cache: {hit_rate:.0f}%"

@app.callback(
    Output("global-legend", "children"),
    [
        Input("question-dropdown", "value"),
        Input("year-range", "value"),
        Input("state-select", "value"),
        Input("options-toggle", "value"),
        Input("main-tabs", "active_tab"),
    ]
)
def update_global_legend(question_text, year_range, selected_states, options_toggle, active_tab):

    # The compare chart has its own legend (questions as lines).
    if active_tab == "tab-compare":
        return ""

    if not question_text:
        return ""

    year_min, year_max = year_range
    states_key = tuple(all_states) if not selected_states else tuple(sorted(selected_states))
    clean_flag = False

    # compute data for all panels
    overall, temporal, state_df, gender, age, race, edu, income = compute_panels_cached(
        question_text, int(year_min), int(year_max), states_key, clean_flag
    )

    # pick the correct dataset depending on active tab
    df_map = {
        "tab-overview": overall,
        "tab-demographics": pd.concat([gender, age, race, edu, income], ignore_index=True),
        "tab-geography": state_df,
        "tab-compare": overall,
        "tab-insights": overall,
    }
    df_source = df_map.get(active_tab, overall)

    if df_source is None or df_source.empty or "Response" not in df_source.columns:
        return ""

    # extract response categories
    responses = sorted(df_source["Response"].dropna().astype(str).unique())

    # get consistent colors using the SAME function charts use
    color_map = get_response_color_map(df_source["Response"])

    # build legend items
    legend_items = []
    for resp in responses:
        color = color_map.get(resp, RESPONSE_COLOR_BASE.get(resp.lower(), "#999"))
        legend_items.append(
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "14px",
                            "height": "14px",
                            "backgroundColor": color,
                            "border": "1px solid #333",
                            "marginRight": "6px",
                            "borderRadius": "2px",
                        }
                    ),
                    html.Span(resp)
                ],
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "marginRight": "18px",
                    "marginBottom": "6px",
                    "fontSize": "0.85rem",
                }
            )
        )

    # wrap in card
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    "Legend — Response Categories",
                    style={"fontWeight": "600", "fontSize": "0.9rem", "marginBottom": "6px"},
                ),
                html.Div(
                    legend_items,
                    style={"display": "flex", "flexWrap": "wrap"},
                ),
            ],
            style={"padding": "8px 14px"},
        ),
        style={"border": "1px solid #ddd", "borderRadius": "6px", "backgroundColor": "#fff"},
    )

@app.callback(
    Output("topic-dropdown", "options"),
    Output("topic-dropdown", "value"),
    Input("class-dropdown", "value"),
)
def update_topic_dropdown(selected_class):
    topic_opts = get_topic_options(selected_class)
    topic_value = topic_opts[0]["value"] if topic_opts else None
    return topic_opts, topic_value


@app.callback(
    Output("question-dropdown", "options"),
    Output("question-dropdown", "value"),
    Input("class-dropdown", "value"),
    Input("topic-dropdown", "value"),
)
def update_question_dropdown(selected_class, selected_topic):
    question_opts = get_question_options(selected_class, selected_topic)
    question_value = question_opts[0]["value"] if question_opts else None
    return question_opts, question_value

# ========== NEW CALLBACKS FOR GRANULARITY CONTROLS ==========

# ========== GRANULARITY CALLBACKS (ADD THESE) ==========


@app.callback(
    Output("income-chart", "figure"),
    Input("income-gran", "value"),
    State("question-dropdown", "value"),
    State("year-range", "value"),
    State("state-select", "value"),
    State("options-toggle", "value"),
    prevent_initial_call=True
)
def update_income_chart(income_gran, question_text, year_range, selected_states, options_toggle):
    if not question_text:
        raise PreventUpdate
    
    year_min, year_max = year_range
    states_key = tuple(all_states) if not selected_states else tuple(sorted(selected_states))
    clean_flag = False
    show_errorbars = "errorbars" in (options_toggle or [])
    
    _, _, _, _, _, _, _, income_df = compute_panels_cached(
        question_text, int(year_min), int(year_max), states_key, clean_flag
    )
    
    if income_df is not None and not income_df.empty and income_gran != 'detailed':
        income_df = group_income_ranges(income_df, level=income_gran)
    
    title = f"By Income ({income_gran.title()} View)"
    return make_bar_figure(income_df, x="Break_Out", y="agg_percent", color="Response",
                          error_col="err", show_errorbars=show_errorbars,
                          title=title, rotate_x=True, height=320)


@app.callback(
    Output("edu-chart", "figure"),
    Input("edu-gran", "value"),
    State("question-dropdown", "value"),
    State("year-range", "value"),
    State("state-select", "value"),
    State("options-toggle", "value"),
    prevent_initial_call=True
)
def update_edu_chart(edu_gran, question_text, year_range, selected_states, options_toggle):
    if not question_text:
        raise PreventUpdate
    
    year_min, year_max = year_range
    states_key = tuple(all_states) if not selected_states else tuple(sorted(selected_states))
    clean_flag = False
    show_errorbars = "errorbars" in (options_toggle or [])
    
    _, _, _, _, _, _, edu_df, _ = compute_panels_cached(
        question_text, int(year_min), int(year_max), states_key, clean_flag
    )
    
    if edu_df is not None and not edu_df.empty and edu_gran != 'standard':
        edu_df = group_education_levels(edu_df, level=edu_gran)
    
    title = f"By Education ({edu_gran.title()} View)"
    return make_bar_figure(edu_df, x="Break_Out", y="agg_percent", color="Response",
                          error_col="err", show_errorbars=show_errorbars,
                          title=title, rotate_x=True, height=320)


@app.callback(
    Output("geo-chart", "figure"),
    Input("geo-gran", "value"),
    State("question-dropdown", "value"),
    State("year-range", "value"),
    State("state-select", "value"),
    State("options-toggle", "value"),
    prevent_initial_call=True
)
def update_geo_chart(geo_gran, question_text, year_range, selected_states, options_toggle):
    if not question_text:
        raise PreventUpdate
    
    year_min, year_max = year_range
    states_key = tuple(all_states) if not selected_states else tuple(sorted(selected_states))
    clean_flag = False
    show_errorbars = "errorbars" in (options_toggle or [])
    
    _, _, state_df, _, _, _, _, _ = compute_panels_cached(
        question_text, int(year_min), int(year_max), states_key, clean_flag
    )
    
    if state_df is not None and not state_df.empty and geo_gran != 'state':
        state_df = aggregate_by_region(state_df, level=geo_gran)
        x_col = 'Region'
    else:
        x_col = 'Locationabbr'
    
    title = f"Geographic Comparison ({geo_gran.title()} Level)"
    return make_bar_figure(state_df, x=x_col, y="agg_percent", color="Response",
                          error_col="err", show_errorbars=show_errorbars,
                          title=title, rotate_x=True, height=500)

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("question-dropdown", "value"),
    Input("year-range", "value"),
    Input("state-select", "value"),
    Input("options-toggle", "value"),
)
def render_tab_content(active_tab, question_text, year_range, selected_states, options_toggle):
    """Render content based on active tab with granularity controls."""
    
    # Set defaults for granularity if None
    
    income_gran = 'standard'
    edu_gran = 'standard'
    geo_gran = 'state'
    
    if question_text is None:
        return html.Div([
            html.Div([
                html.I(className="bi bi-arrow-left", style={'fontSize': '64px', 'color': '#ccc'}),
                html.H4("Select a question from the sidebar to begin", className="mt-3 text-muted")
            ], className="text-center", style={'paddingTop': '200px'})
        ])
    
    year_min, year_max = year_range
    states_key = tuple(all_states) if not selected_states else tuple(sorted(selected_states))
    clean_flag = False
    show_errorbars = "errorbars" in (options_toggle or [])
    
    # Get data
    (overall_df, temporal_df, state_df, gender_df, age_df, race_df, edu_df, income_df) = compute_panels_cached(
        question_text, int(year_min), int(year_max), states_key, clean_flag
    )
    
    # Apply granularity grouping
    
    if income_df is not None and not income_df.empty and income_gran != 'detailed':
        income_df = group_income_ranges(income_df, level=income_gran)
    
    if edu_df is not None and not edu_df.empty and edu_gran != 'standard':
        edu_df = group_education_levels(edu_df, level=edu_gran)
    
    # Apply geographic grouping
    if state_df is not None and not state_df.empty and geo_gran != 'state':
        state_df_grouped = aggregate_by_region(state_df, level=geo_gran)
        x_col = 'Region'
    else:
        state_df_grouped = state_df
        x_col = 'Locationabbr'
    
    # TAB 1: OVERVIEW
    if active_tab == "tab-overview":
        if overall_df is not None and not overall_df.empty:
            overall_df_plot = overall_df.copy()
            overall_df_plot["overall_label"] = "Overall"
        else:
            overall_df_plot = None
            
        fig_overall = make_bar_figure(
            overall_df_plot, x="overall_label", y="agg_percent", color="Response",
            error_col="err", show_errorbars=show_errorbars,
            title="National Prevalence", height=500
        )
        
        fig_temporal = make_line_figure(
            temporal_df, x="Year", y="agg_percent", color="Response",
            title="Trends Over Time", x_label="Year", height=500
        )
        
        return dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_overall, config={'displayModeBar': False})], md=6),
            dbc.Col([dcc.Graph(figure=fig_temporal, config={'displayModeBar': False})], md=6),
        ])
    
    # TAB 2: DEMOGRAPHICS
    elif active_tab == "tab-demographics":
        # Granularity controls
        controls = dbc.Card([
            dbc.CardBody([
                html.H6("🔍 Detail Level Controls", className="mb-2", style={'fontWeight': '600'}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Income Levels:", style={'fontSize': '0.8rem', 'fontWeight': '600'}),
                        dcc.RadioItems(
                            id='income-gran',
                            options=[
                                {'label': ' Simple (3)', 'value': 'simple'},
                                {'label': ' Standard (5)', 'value': 'standard'},
                                {'label': ' Detailed (7)', 'value': 'detailed'},
                            ],
                            value=income_gran,
                            inline=True,
                            style={'fontSize': '0.8rem'}
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Education:", style={'fontSize': '0.8rem', 'fontWeight': '600'}),
                        dcc.RadioItems(
                            id='edu-gran',
                            options=[
                                {'label': ' Simple (3)', 'value': 'simple'},
                                {'label': ' Standard (4)', 'value': 'standard'},
                            ],
                            value=edu_gran,
                            inline=True,
                            style={'fontSize': '0.8rem'}
                        )
                    ], md=4),
                ])
            ], style={'padding': '12px'})
        ], className="mb-2")
        
        fig_gender = make_bar_figure(gender_df, x="Break_Out", y="agg_percent", color="Response",
                                     error_col="err", show_errorbars=show_errorbars,
                                     title="By Gender", rotate_x=False, height=320)
        
        age_title = f"By Age Group "
        fig_age = make_bar_figure(age_df, x="Break_Out", y="agg_percent", color="Response",
                                  error_col="err", show_errorbars=show_errorbars,
                                  title=age_title, rotate_x=True, height=320)
        
        fig_race = make_bar_figure(race_df, x="Break_Out", y="agg_percent", color="Response",
                                   error_col="err", show_errorbars=show_errorbars,
                                   title="By Race/Ethnicity", rotate_x=True, height=320)
        
        edu_title = f"By Education ({edu_gran.title()} View)"
        fig_edu = make_bar_figure(edu_df, x="Break_Out", y="agg_percent", color="Response",
                                  error_col="err", show_errorbars=show_errorbars,
                                  title=edu_title, rotate_x=True, height=320)
        
        income_title = f"By Income ({income_gran.title()} View)"
        fig_income = make_bar_figure(income_df, x="Break_Out", y="agg_percent", color="Response",
                                     error_col="err", show_errorbars=show_errorbars,
                                     title=income_title, rotate_x=True, height=320)
        
        return html.Div([
            controls,
            dbc.Row([
                dbc.Col([dcc.Graph(id="gender-chart",figure=fig_gender, config={'displayModeBar': False})], md=4),
                dbc.Col([dcc.Graph(id="age-chart",figure=fig_age, config={'displayModeBar': False})], md=4),
                dbc.Col([dcc.Graph(id="race-chart",figure=fig_race, config={'displayModeBar': False})], md=4),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([dcc.Graph(id="edu-chart",figure=fig_edu, config={'displayModeBar': False})], md=6),
                dbc.Col([dcc.Graph(id="income-chart",figure=fig_income, config={'displayModeBar': False})], md=6),
            ]),
        ])
    
    # TAB 3: GEOGRAPHY
    elif active_tab == "tab-geography":
        geo_control = dbc.Card([
            dbc.CardBody([
                html.Label("🌎 Geographic Detail Level:", style={'fontSize': '0.85rem', 'fontWeight': '600', 'marginRight': '15px'}),
                dcc.RadioItems(
                    id='geo-gran',
                    options=[
                        {'label': ' Regions (4)', 'value': 'region'},
                        {'label': ' Divisions (9)', 'value': 'division'},
                        {'label': ' States (50+)', 'value': 'state'},
                    ],
                    value=geo_gran,
                    inline=True,
                    style={'fontSize': '0.85rem'}
                )
            ], style={'padding': '12px'})
        ], className="mb-2")
        
        geo_title = f"Geographic Comparison ({geo_gran.title()} Level)"
        fig_state = make_bar_figure(state_df_grouped, x=x_col, y="agg_percent", color="Response",
                                    error_col="err", show_errorbars=show_errorbars,
                                    title=geo_title, rotate_x=True, height=500)
        
        fig_map = make_choropleth_figure(state_df, title="State-Level Distribution Map", 
                                        target_response="yes", height=500)
        
        return html.Div([
            geo_control,
            dbc.Row([
                dbc.Col([dcc.Graph(id="geo-chart",figure=fig_state, config={'displayModeBar': False})], md=6),
                dbc.Col([dcc.Graph(figure=fig_map, config={'displayModeBar': False})], md=6),
            ])
        ])
    
   # TAB 4: COMPARE
    elif active_tab == "tab-compare":
        compare_selector = dbc.Card([
            dbc.CardBody([
                html.Label(
                    "Compare primary question with:",
                    style={'fontSize': '0.85rem', 'fontWeight': '600'}
                ),
                dcc.Dropdown(
                    id="compare-question",
                    options=compare_question_options,
                    value=None,
                    placeholder="Select another question...",
                    style={'fontSize': '0.85rem'},
                    optionHeight=60
                ),
                html.Small(
                    "This tab shows how each question’s overall percentage changed from the first year in the range to the last year.",
                    className="text-muted"
                )
            ], style={'padding': '12px'})
        ], className="mb-2")
        
        return html.Div([
            compare_selector,
            html.Div(id="compare-plot-container")
        ])
    
    # TAB 5: INSIGHTS & DATA
    elif active_tab == "tab-insights":
        insights_text = generate_insight_summary(overall_df, temporal_df, state_df, 
                                                 gender_df, age_df, race_df, edu_df, income_df)
        
        insights_card = dbc.Card([
            dbc.CardHeader(html.H5("💡 Key Insights", className="mb-0", style={'fontWeight': '600'})),
            dbc.CardBody([
                dcc.Markdown(insights_text, style={'whiteSpace': 'pre-line', 'lineHeight': '1.8', 'fontSize': '0.9rem'})
            ])
        ], className="mb-3")
        
        ci_selector = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Show Confidence Intervals for:", style={'fontSize': '0.85rem', 'fontWeight': '600'}),
                    ], md=3),
                    dbc.Col([
                        dcc.Dropdown(
                            id="ci-breakdown-selector",
                            options=[
                                {"label": "Overall Results", "value": "overall"},
                                {"label": "By Gender", "value": "gender"},
                                {"label": "By Age Group", "value": "age"},
                                {"label": "By Race/Ethnicity", "value": "race"},
                                {"label": "By Education Level", "value": "education"},
                                {"label": "By Income Level", "value": "income"},
                            ],
                            value="overall",
                            style={'fontSize': '0.85rem'}
                        ),
                    ], md=9),
                ])
            ], style={'padding': '12px'})
        ], className="mb-2")
        
        return html.Div([
            insights_card,
            ci_selector,
            html.Div(id="ci-table-container")
        ])
    
    return html.Div("Select a tab")


@app.callback(
    Output("ci-table-container", "children"),
    Input("ci-breakdown-selector", "value"),
    Input("question-dropdown", "value"),
    Input("year-range", "value"),
    Input("state-select", "value"),
    Input("options-toggle", "value"),
    # State("age-gran", "value"),              # ← CHANGED TO State
    # State("income-gran", "value"),           # ← CHANGED TO State
    # State("edu-gran", "value"),              # ← CHANGED TO State
)
def update_ci_table(selected_breakdown, question_text, year_range, selected_states, options_toggle,):
    """Update CI table based on selected breakdown with granularity applied."""
    if question_text is None or selected_breakdown is None:
        return html.Div("No data available", className="text-muted text-center p-3")
    
    # # Set defaults
    # age_gran = age_gran or 'standard'
    # income_gran = income_gran or 'standard'
    # edu_gran = edu_gran or 'standard'
    
    year_min, year_max = year_range
    states_key = tuple(all_states) if not selected_states else tuple(sorted(selected_states))
    clean_flag = False
    
    (overall_df, _, _, gender_df, age_df, race_df, edu_df, income_df) = compute_panels_cached(
        question_text, int(year_min), int(year_max), states_key, clean_flag
    )
    
    # Apply grouping to match what's shown in charts
    # if age_df is not None and not age_df.empty and age_gran != 'detailed':
    #     age_df = group_age_ranges(age_df, level=age_gran)
    
    # if income_df is not None and not income_df.empty and income_gran != 'detailed':
    #     income_df = group_income_ranges(income_df, level=income_gran)
    
    # if edu_df is not None and not edu_df.empty and edu_gran != 'standard':
    #     edu_df = group_education_levels(edu_df, level=edu_gran)
    
    table_df = build_ci_table(selected_breakdown, overall_df, gender_df, age_df, race_df, edu_df, income_df)
    
    if table_df.empty:
        return html.Div("No data available", className="text-muted text-center p-3")
    
    return dbc.Card([
        dbc.CardBody([
            dash_table.DataTable(
                data=table_df.to_dict('records'),
                columns=[{"name": col, "id": col} for col in table_df.columns],
                style_table={'overflowX': 'auto', 'maxHeight': '450px', 'overflowY': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px',
                    'fontSize': '0.9rem',
                    'fontFamily': 'Arial, sans-serif',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_header={
                    'backgroundColor': COLORS['primary'],
                    'color': 'white',
                    'fontWeight': 'bold',
                    'fontSize': '0.9rem',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgba(248, 249, 250, 0.5)'
                    },
                    {
                        'if': {'column_id': 'Response'},
                        'fontWeight': '500'
                    }
                ],
                page_size=30,
                sort_action='native',
                filter_action='native',
            )
        ])
    ])


@app.callback(
    Output("compare-plot-container", "children"),
    Input("compare-question", "value"),
    Input("question-dropdown", "value"),
    Input("year-range", "value"),
    Input("state-select", "value"),
    Input("options-toggle", "value"),
)
def update_compare_plot(q2, q1, year_range, selected_states, options_toggle):
    """Compare tab: show line plot + numeric summary for two questions."""
    # Need a primary question
    if q1 is None:
        return html.Div(
            [
                html.I(className="bi bi-arrow-left", style={'fontSize': '48px', 'color': '#ccc'}),
                html.P("Select a primary question from the sidebar first.", className="text-muted mt-3"),
            ],
            className="text-center",
            style={"paddingTop": "150px"},
        )

    # Need a second question
    if q2 is None:
        return html.Div(
            [
                html.I(className="bi bi-arrow-up", style={'fontSize': '48px', 'color': '#ccc'}),
                html.P("Select a second question above to compare.", className="text-muted mt-3"),
            ],
            className="text-center",
            style={"paddingTop": "150px"},
        )

    if q1 == q2:
        return dbc.Alert(
            "⚠️ Please select two different questions to compare.",
            color="warning",
            className="mt-3",
        )

    year_min, year_max = year_range
    states_key = tuple(all_states) if not selected_states else tuple(sorted(selected_states))
    clean_flag = False  # we removed 'clean responses'

    # Get temporal panels (overall by year) for both questions
    _, temp1, *_ = compute_panels_cached(q1, int(year_min), int(year_max), states_key, clean_flag)
    _, temp2, *_ = compute_panels_cached(q2, int(year_min), int(year_max), states_key, clean_flag)

    if temp1 is None or temp1.empty or temp2 is None or temp2.empty:
        return dbc.Alert(
            "No temporal data available for this comparison.",
            color="info",
            className="mt-3",
        )

    # Use AUTO mode in extract_main_series (primary/best response)
    s1 = extract_main_series(temp1, q1, target_response="__auto__")
    s2 = extract_main_series(temp2, q2, target_response="__auto__")

    series_list = [s for s in (s1, s2) if s is not None and not s.empty]
    if not series_list:
        return dbc.Alert(
            "Unable to summarize change over time for these questions.",
            color="info",
            className="mt-3",
        )

    # ---------- Build line figure (levels view) ----------
    combined = pd.concat(series_list, ignore_index=True)
    fig = make_line_figure(
        combined,
        x="Year",
        y="agg_percent",
        color="QuestionLabel",
        title="Temporal Comparison of Two Questions",
        x_label="Year",
        height=550,
    )
    # Ensure legend is visible (override global showlegend=False)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # ---------- Build numeric summary ----------
    summary_items = []

    for label, series in ((q1, s1), (q2, s2)):
        if series is None or series.empty:
            continue

        series_sorted = series.sort_values("Year")
        first_year = int(series_sorted["Year"].iloc[0])
        last_year = int(series_sorted["Year"].iloc[-1])
        first_val = float(series_sorted["agg_percent"].iloc[0])
        last_val = float(series_sorted["agg_percent"].iloc[-1])
        delta = last_val - first_val

        trend_symbol = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
        trend_color = "#06A77D" if delta > 0 else ("#C73E1D" if delta < 0 else "#6c757d")

        summary_items.append(
            html.Div(
                [
                    html.Div(
                        label,
                        style={"fontWeight": "600", "fontSize": "0.85rem"},
                    ),
                    html.Div(
                        f"{first_year}: {first_val:.1f}% → {last_year}: {last_val:.1f}% "
                        f"({trend_symbol} {delta:+.1f} pts)",
                        style={"fontSize": "0.8rem", "color": trend_color},
                    ),
                ],
                style={"marginBottom": "8px"},
            )
        )

    if not summary_items:
        summary_card = dbc.Alert(
            "No summary could be computed for these questions.",
            color="info",
            className="mt-3",
        )
    else:
        summary_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        "Summary of change over time",
                        style={"fontWeight": "600", "fontSize": "0.9rem", "marginBottom": "4px"},
                    ),
                    html.Div(
                        summary_items,
                        style={"fontSize": "0.8rem"},
                    ),
                ]
            ),
            className="mt-3",
        )

    return html.Div(
        [
            graph,
            summary_card,
        ]
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting dashboard server...")
    print("📊 Open: http://127.0.0.1:8050/")
    print("="*60 + "\n")
    app.run(debug=True)