import pandas as pd
import numpy as np

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

import statsmodels.formula.api as smf

colors = px.colors.qualitative.D3
metrics = ['registered', 'casual', 'total']


def parse_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "dteday": "date",
        "cnt": "total", 
        "hr": "hour", 
        "mnth": "month", 
        "yr": "year",
        "weathersit": "weather_condition"
    })

    # dates
    df['date'] = pd.to_datetime(df['date'])
    df["year_month"] = df["date"].dt.strftime("%b%Y")
    df["day_of_week"] = df["date"].dt.strftime("%a")

    # weather
    df['weather_condition'] = df['weather_condition'].map({1: "clear", 2: "cloudy", 3: "light", 4: "heavy"})
    df['weather_condition'] = pd.Categorical(
        df['weather_condition'], 
        categories=['clear', 'cloudy', 'light', 'heavy'],
        ordered=True
    )

    # season
    df['season'] = df['season'].map({2: "spring", 3: "summer", 4: "fall", 1: "winter"})
    df['season'] = pd.Categorical(
        df['season'],
        categories=["spring", "summer", "fall", "winter"],
        ordered=True,
    )

    # hour grouping
    if 'hour' in df:
        hour_categories = [
            (0, 6, 'early_morning'),
            (7, 9, 'morning_commute'),
            (10, 15, 'off_peak'),
            (16, 19, 'afternoon_commute'),
            (20, 23, 'night'),
        ]
        bins= [0] + [e for _, e, _ in hour_categories]
        labels = [cat for _, _, cat in hour_categories]
        df['hour_group'] = pd.cut(df['hour'], bins=bins, labels=labels, include_lowest=True)
        df['is_commute'] = df['hour_group'].str.contains('commute')

    # is bad weather?
    df['is_rainy'] = df['weather_condition'].isin(['light', 'heavy'])
    df['is_cold'] = df['temp'].le(df['temp'].quantile(0.2))
    df['bad_weather'] = df['is_rainy'] | df['is_cold']
    
    return df


# EDA Plots
def plot_usage_per_weather_condition(hourly: pd.DataFrame, metric: str = "total"):
    
    df = hourly.copy()
    agg = df.groupby('weather_condition')[metric].mean().reset_index()
    global_avg = df[metric].mean()
    
    fig = px.bar(
        agg,
        x="weather_condition",
        y=metric,
        color="weather_condition",
        color_discrete_sequence=px.colors.sequential.Blues[3::2],
    )
    
    fig.add_trace(go.Scatter(
        x=[agg['weather_condition'].iat[0], agg['weather_condition'].iat[-1]],
        y=2 * [global_avg],
        mode="lines+text",
        text=["", "average"],
        line=dict(color='black', dash='dot', width=3),
        textposition="top left",
        name="global average",
    ))
    
    fig.update_traces(dict(marker=dict(line=dict(width=2, color='lightgray'))))
    
    fig.update_layout(
        title="<b>Average Hourly Bike Share per Weather Condition</b>",
        title_x=0.5,
        width=800,
        height=400,
        showlegend=False,
    )
    
    fig.update_yaxes(title=None)
    
    return fig


def plot_usage_per_temperature(hourly: pd.DataFrame, metric: str = "total"):
    df = hourly.copy()
    df['temp_bucket'] = pd.qcut(df['temp'], 10)
    global_avg = df[metric].mean()
    
    agg = df.groupby('temp_bucket')[metric].mean().to_frame(metric)
    agg['percentile'] = np.arange(10) / 10
    agg['bucket'] = [f"{10*x:.0f}-{10*(x+1):.0f}%" for x in range(10)]
    
    fig = px.bar(
        agg,
        x='bucket',
        y=metric,
        color='percentile',
        color_continuous_scale=px.colors.sequential.RdBu_r,
    )
    
    fig.add_trace(go.Scatter(
        x=[agg['bucket'].iat[0], agg['bucket'].iat[-1]],
        y=2 * [global_avg],
        mode="lines+text",
        text=["average", ""],
        line=dict(color='black', dash='dot', width=3),
        textposition="top right",
        name="global average",
    ))
    
    fig.update_traces(dict(marker=dict(line=dict(width=2, color='lightgray'))))
    
    fig.update_layout(
        title="<b>Average Hourly Bike Share per Temperature</b>",
        title_x=0.5,
        width=800,
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
    )
    
    fig.update_yaxes(title=None)
    
    fig.update_xaxes(title="temperature percentile")
    
    return fig


def table_bad_weather_recap(hourly: pd.DataFrame, metric: str = 'total'):
    df = hourly.copy()
    
    agg = df.groupby('bad_weather').agg(n_hours=(metric, 'count'), avg_demand_per_hour=(metric, 'mean'))
    agg = agg.rename_axis(index=None)
    agg = agg.rename(index={False: "good weather", True: "bad weather"})
    agg["hours_pct"] = agg["n_hours"] / agg["n_hours"].sum()
    agg = agg[["n_hours", "hours_pct", "avg_demand_per_hour"]]
    
    styler = agg.style.format({"n_hours": "{:,.0f}", "hours_pct": "{:.0%}", "avg_demand_per_hour": "{:.0f}"})

    return styler


# Naive Plots
def plot_naive_estimate(hourly: pd.DataFrame, metric: str = 'total'):
    df = hourly.copy()
    
    agg = df.groupby('bad_weather')[metric].mean()
    agg = agg.rename(index={False: "good weather", True: "bad weather"})
    good_weather, bad_weather = agg
    agg = agg.reset_index()
    
    fig = px.bar(
        agg, 
        x="bad_weather", 
        y=metric, 
        color="bad_weather",
        color_discrete_sequence=px.colors.sequential.Blues[4::3],
    )
    
    fig.update_layout(
        title=f"<b>Naive Estimate of Bad Weather Effect</b>",
        title_x=0.5,
        width=600,
        height=400,
        showlegend=False,
    )
    
    # drop arrow
    fig.add_annotation(
        x=0.45,
        ax=0.45,
        y=bad_weather,
        ay=good_weather,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowside="end",
        arrowwidth=2,
        
    )
    
    fig.add_annotation(
        x=0.55,
        y=(good_weather+bad_weather)/2,
        text=f"<b>{bad_weather/good_weather-1:+.0%}</b>",
        showarrow=False,
    )
    
    fig.update_yaxes(title=None, range=(0, 1.2 * agg[metric].max()))
    
    fig.update_xaxes(title=None)
    
    return fig


def add_grouping_annotations(fig, bins, labels, line_spacing: float = 0.1):
    y_values = []
    for trace in fig.data:
        if hasattr(trace, "y") and trace.y is not None:
                y_values.extend(trace.y)
    
        y_max = max(y_values) if y_values else 1
        y_bracket = y_max * 1.1
        tick_height = y_max * 0.05
        line_intervals = [(start + line_spacing, end - line_spacing) for start, end in zip(bins[:-1], bins[1:])]
    
    for (start, end), label in zip(line_intervals, labels):
        fig.add_shape(
            type="line",
            x0=start,
            x1=end,
            y0=y_bracket,
            y1=y_bracket,
            line=dict(color="black", width=2)
        )

        fig.add_shape(
            type="line",
            x0=start,
            x1=start,
            y0=y_bracket,
            y1=y_bracket - tick_height,
            line=dict(color="black", width=2)
        )

        fig.add_shape(
            type="line",
            x0=end,
            x1=end,
            y0=y_bracket,
            y1=y_bracket - tick_height,
            line=dict(color="black", width=2)
        )

        fig.add_annotation(
            x=(start + end) / 2,
            y=y_bracket + tick_height,
            text=label,
            showarrow=False
        )

    current_margin = fig.layout.margin.t if fig.layout.margin and fig.layout.margin.t else 0
    fig.update_layout(margin=dict(t=max(current_margin, 100)))


def plot_hourly_usage(hourly: pd.DataFrame, metric: str = 'total'):
    df = hourly.copy()
    
    hour_categories = [
        (0, 6, 'early_morning'),
        (7, 9, 'morning_commute'),
        (10, 15, 'off_peak'),
        (16, 19, 'afternoon_commute'),
        (20, 23, 'night'),
    ]
    bins= [0] + [e for _, e, _ in hour_categories]
    labels = [cat for _, _, cat in hour_categories]
    
    histogram_data = df.groupby('hour')[metric].mean()
    histogram_data = histogram_data.to_frame(name='count')
    histogram_data['category'] = pd.cut(histogram_data.index, bins=bins, labels=labels, include_lowest=True)
        
    fig = px.bar(
        histogram_data,
        y='count',
        color='category',
        color_discrete_sequence=colors,
    )
    
    add_grouping_annotations(
        fig,
        bins=[x + 0.5 if x > 0 else x for x in bins],
        labels=labels,
    )
    
    fig.update_layout(
        showlegend=False,
        width=800,
        height=400,
        title="<b>Average Hourly Bike Share</b>",
        title_x=0.5,
        margin=dict(t=40),
    )
    
    fig.update_yaxes(title=None)
    
    fig.update_xaxes(title="hour")
    
    return fig


def plot_usage_per_month(hourly: pd.DataFrame, metric: str = 'total'):

    df = hourly.copy()
    
    agg = df.groupby(pd.Grouper(freq="MS", key="date"))[[metric, 'temp']].mean()
    
    fig = px.bar(
        agg, 
        y=metric, 
        color='temp', 
        color_continuous_scale=px.colors.sequential.RdBu_r,
    )
    
    fig.update_traces(dict(marker=dict(line=dict(width=2, color='lightgray'))))
    
    fig.update_layout(
        title="<b>Average Hourly Bike Share per Month</b>",
        title_x=0.5,
        width=800,
        height=400,
        coloraxis_showscale=False,
    )
    
    fig.update_yaxes(title=None)
    
    fig.update_xaxes(title=None)

    return fig


# OLS Fit
def get_summary_statmodel_table(model) -> pd.DataFrame:
    df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
        "ci_lower": model.conf_int()[0],
        "ci_upper": model.conf_int()[1]
    })
    
    df = df.reset_index().rename(columns={"index": "variable"})

    return df

def fit_causal_inference_model(hourly: pd.DataFrame, formula: str, return_model: bool = False) -> pd.DataFrame:
    df = hourly.copy()
    for target in metrics:
        log_target = f"log_{target}"
        df[log_target] = np.nan
        df.loc[df[target].gt(0), log_target] = np.log(df.loc[df[target].gt(0), target])

        
    model = smf.ols(formula, data=df).fit()

    if return_model:
        return model
        
    return get_summary_statmodel_table(model)


def get_estimate_and_ci(hourly: pd.DataFrame, formula: str, log_transform: bool = False) -> pd.Series:
    model = fit_causal_inference_model(hourly, formula)
    coefs = model.loc[1, ["coef", "ci_lower", "ci_upper"]]
    
    if log_transform:
        coefs = np.exp(coefs) - 1
        
    return pd.Series(coefs)


# Causal Inference
def plot_causal_estimate_against_naive(hourly: pd.DataFrame):
    model_formula = 'log_total ~ bad_weather + C(hour) + C(year_month)'
    naive_formula = 'log_total ~ bad_weather'
    
    model = fit_causal_inference_model(hourly, model_formula)
    estimate, lower, upper = np.exp(model.loc[1, ['coef', 'ci_lower', 'ci_upper']]) - 1
    
    naive_model = fit_causal_inference_model(hourly, naive_formula)
    naive_estimate, naive_lower, naive_upper = np.exp(naive_model.loc[1, ['coef', 'ci_lower', 'ci_upper']]) - 1
    
    agg = hourly.groupby('bad_weather')['total'].mean()
    manual_estimate, manual_lower, manual_upper = agg[True] / agg[False] - 1, np.nan, np.nan
    
    labels = ["Causal (FE)", "Log Naive", "Mean Ratio"]
    estimates = np.array([estimate, naive_estimate, manual_estimate])
    lowers = np.array([lower, naive_lower, manual_lower])
    uppers = np.array([upper, naive_upper, manual_upper])
    error_plus = uppers - estimates
    error_minus = estimates - lowers
    
    fig = go.Figure([
        go.Bar(
            x=labels,
            y=estimates,
            marker_color=px.colors.sequential.Greens[4::2],
            error_y=dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                thickness=1.5,
                width=20,
            ),
        )
    ])
    
    fig.update_layout(
        title="<b>Causal Impact of Bad Weather on Demand</b>",
        title_x=0.5,
        width=600,
        height=400,
        margin=dict(t=50),
    )
    
    fig.update_yaxes(tickformat="+0.0%")
    
    fig.update_xaxes(title="model_name")

    return fig


def plot_causal_effect_per_model(hourly: pd.DataFrame):
    df = hourly.copy()
    
    formulas = {
        "naive": "log_total ~ bad_weather",
        "month": "log_total ~ bad_weather + C(year_month)",
        "hour_groups": "log_total ~ bad_weather + C(hour_group) + C(year_month)",
        "hour": "log_total ~ bad_weather + C(hour) + C(year_month)",
        "day_of_week": "log_total ~ bad_weather + C(hour) + C(year_month) + C(workingday) + C(day_of_week)",
    }
    estimates = pd.DataFrame.from_dict(
        {k: get_estimate_and_ci(df, v, log_transform=True) for k, v in formulas.items()},
        orient='index'
    )
    estimates['error_plus'] = estimates['ci_upper'] - estimates['coef']
    estimates['error_minus'] = estimates['coef'] - estimates['ci_lower']
    
    fig = go.Figure([go.Bar(
        x=estimates.index,
        y=estimates['coef'],
        marker_color=px.colors.sequential.Greens_r,
        error_y=dict(
            type="data",
            symmetric=False,
            array=estimates['error_plus'],
            arrayminus=estimates['error_minus'],
            thickness=1.5,
            width=5,
        ),
    )])
    
    fig.update_layout(
        title="<b>Causal Impact per Model</b>",
        title_x=0.5,
        width=600,
        height=400,
        margin=dict(t=50),
    )
    
    fig.update_yaxes(tickformat="+0.0%")
    
    fig.update_xaxes(title="complexity (simple->complex)")

    return fig


def plot_causal_effect_per_hour_group(hourly: pd.DataFrame):
    df = hourly.copy()
    
    model_formula = 'log_total ~ bad_weather + C(hour) + C(year_month)'
    
    # 1 model with interactions would have similar results
    # for simplicity use existing function and groupby
    estimates = pd.DataFrame.from_dict(
        {
            hour_group: get_estimate_and_ci(sub_df, model_formula, log_transform=True)
            for hour_group, sub_df in df.groupby('hour_group')
        },
        orient='index',
    )
    
    estimates.loc["global"] = get_estimate_and_ci(df, model_formula, log_transform=True)
    estimates['error_plus'] = estimates['ci_upper'] - estimates['coef']
    estimates['error_minus'] = estimates['coef'] - estimates['ci_lower']
    
    marker_colors = dict(zip(estimates.index, colors))
    marker_colors |= {"global": "lightgray"}
    marker_colors = list(marker_colors.values())
    
    fig = go.Figure([go.Bar(
        x=estimates.index,
        y=estimates['coef'],
        marker_color=marker_colors,
        error_y=dict(
            type="data",
            symmetric=False,
            array=estimates['error_plus'],
            arrayminus=estimates['error_minus'],
            thickness=1.5,
            width=5,
        ),
    )])
    
    fig.update_layout(
        title="<b>Effect of Bad Weather per Hour Group</b>",
        title_x=0.5,
        width=800,
        height=400,
        margin=dict(t=50),
    )
    
    fig.update_yaxes(tickformat="+0.0%")
    
    fig.update_xaxes(title=None)

    return fig


def plot_causal_effect_per_user_type(hourly: pd.DataFrame):
    df = hourly.copy()
    
    formulas = {
        "global": "log_total ~ bad_weather + C(hour) + C(year_month)",
        "registered": "log_registered ~ bad_weather + C(hour) + C(year_month) + C(workingday)",
        "casual": "log_casual ~ bad_weather + C(hour) + C(year_month) + C(workingday)",
    }
    estimates = pd.DataFrame.from_dict(
        {k: get_estimate_and_ci(df, v, log_transform=True) for k, v in formulas.items()},
        orient='index'
    )
    estimates['error_plus'] = estimates['ci_upper'] - estimates['coef']
    estimates['error_minus'] = estimates['coef'] - estimates['ci_lower']
    
    fig = go.Figure([go.Bar(
        x=estimates.index,
        y=estimates['coef'],
        marker_color=['lightgray', px.colors.qualitative.D3[2], px.colors.qualitative.D3[3]],
        error_y=dict(
            type="data",
            symmetric=False,
            array=estimates['error_plus'],
            arrayminus=estimates['error_minus'],
            thickness=1.5,
            width=5,
        ),
    )])
    
    fig.update_layout(
        title="<b>Effect of Bad Weather per User Type</b>",
        title_x=0.5,
        width=600,
        height=400,
        margin=dict(t=50),
    )
    
    fig.update_yaxes(tickformat="+0.0%")
    
    fig.update_xaxes(title="user type")

    return fig