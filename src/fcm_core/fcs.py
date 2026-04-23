import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fcs_time_scatter_plot(
    selected_df: pd.DataFrame,
    df: pd.DataFrame,
    signal_col: str = "PKH26",
    ref_exp_id: str = "02",
    exclude_thresh_line: list = ["01"],
):
    """Plot histogram distributions of signal groups over time.

    Creates stacked subplots — one per experiment — showing the log-scaled
    count distribution for Control, Low, High, and (optionally) Parental
    sample groups. Dashed vertical reference lines mark the median Low- and
    High-uptake thresholds derived from a chosen reference experiment.

    Args:
        selected_df: Filtered dataframe containing the rows
            to plot. Must have columns ``Experimental_ID``, ``Sample``, and
            the column specified by ``signal_col``.
        df: Full dataframe used to compute reference medians
            and subplot titles. Must have columns ``Experimental_ID``,
            ``Sample``, ``Day``, and the column specified by ``signal_col``.
        signal_col: Name of the fluorescence signal column to plot.
            Defaults to ``"PKH26"``.
        ref_exp_id: Experimental ID whose Low/High medians are used as
            the dashed reference lines. Defaults to ``"02"``.
        exclude_thresh_line: List of Experimental IDs whose thresholds should 
            be excluded from the reference lines. Defaults to ``["01"]``.

    Returns:
        go.Figure: Plotly figure with one subplot row per experiment.
    """
    len_exp = len(selected_df["Experimental_ID"].unique())
    # define the threshold dashed lines on the graph
    low_ref = df[signal_col].loc[
        (df["Experimental_ID"] == ref_exp_id) & (df["Sample"] == "Low")
    ].median()  # median of low-uptake signal from the reference experiment
    high_ref = df[signal_col].loc[
        (df["Experimental_ID"] == ref_exp_id) & (df["Sample"] == "High")
    ].median()  # median of high-uptake signal from the reference experiment
    # Make a list of titles that should be added to the subplot:
    subplot_titles_list = tuple(selected_df["Day"].unique())

    # Make the subplots
    fig = make_subplots(
        rows=len_exp, cols=1, subplot_titles=subplot_titles_list
    )
    i = 1  # number of first row

    for exp in selected_df["Experimental_ID"].unique():
        exp_df = selected_df.loc[selected_df["Experimental_ID"] == exp]
        # The data for each group
        ctrl_data = exp_df[signal_col].loc[exp_df["Sample"] == "Control"]
        low_data = exp_df[signal_col].loc[
            exp_df["Sample"] == "Low"
        ]  # Data is exponential itself and does not need any modifications.
        high_data = exp_df[signal_col].loc[exp_df["Sample"] == "High"]
        parental_data = exp_df[signal_col].loc[
            exp_df["Sample"] == "Parental"
        ]  # Data from the native DU145 (at plateau stage)

        # log-scaled bins. The first two values are the exponents. This numpy
        # uses the base 10 to calculate the argument (main value). The same
        # bins should be used for plotting each group.
        bins = np.logspace(0, 6, 256)

        # Calculate the count falling in each bin for each group. Ignore the
        # generated bin_edges of np.histogram.
        hist_ctrl, _ = np.histogram(ctrl_data, bins=bins)  # Control group.
        hist_low, _ = np.histogram(low_data, bins=bins)  # Low-uptake group.
        hist_high, _ = np.histogram(high_data, bins=bins)  # High_uptake group.
        hist_parental, _ = np.histogram(
            parental_data, bins=bins
        )  # DU145 native parental control

        # Make a plotly graph with different traces. In these graphs, the
        # opacity of the filled area can only be controlled via rgba. The
        # fourth letter of rgba, is alpha (opacity).
        # The forth value is between 0 and 1.
        # Go to the "Color Models" of the webpage
        # https://www.flatuicolorpicker.com/ for finding the rgb of any color.
        fig.add_trace(
            go.Scatter(
                x=bins,
                y=hist_ctrl,
                name="Control",
                line_color="black",
                fillcolor="rgba(36, 37, 42, 0.5)",
                fill="tozeroy",
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bins,
                y=hist_low,
                name="Low",
                line_color="green",
                fillcolor="rgba(168, 229, 175, 0.5)", # "rgba(22, 69, 62, 0.5)"
                fill="tozeroy",
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bins,
                y=hist_high,
                name="High",
                line_color="red",
                fillcolor="rgba(214, 96, 77, 0.5)",
                fill="tozeroy",
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        if "Parental" in list(exp_df["Sample"].unique()):
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=hist_parental,
                    name="Parental",
                    line_color="blue",
                    fillcolor="rgba(45, 85, 255, 0.5)",
                    fill="tozeroy",
                    showlegend=False,
                ),
                row=i,
                col=1,
            )

        # Change the scale of the xaxis from linear to log.
        fig.update_xaxes(type="log")

        # Make the box borders around the graph to look better.
        fig.update_xaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_text="",
            title_font_size=20,
            tickfont_size=20,
            showgrid=False,  # Disable gridlines
        )
        fig.update_yaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            range=[0, 700],
            title_text="Count",
            title_font_size=20,
            tickfont_size=20,
            showgrid=False,  # Disable gridlines
        )
        if exp not in(exclude_thresh_line):
            # The reference line for the low uptake
            fig.add_vline(
                x=low_ref,
                line_dash="dash",
                line_color="rgba(22, 69, 62, 0.5)",
                row=i,
                col=1,
            )
            # The reference line for the high uptake
            fig.add_vline(
                x=high_ref,
                line_dash="dash",
                line_color="rgba(255, 0, 0, 0.5)",
                row=i,
                col=1,
            )

        i += 1  # go to the next row

    # Add title
    title = "Distribution of low and high-uptake groups over time"
    fig.update_layout(
        title=title, title_x=0.5
    )  # Add title and place it in the middle of the canvas
    fig.update_layout(
        height=500, width=600, plot_bgcolor="white", title_font_size=20
    )
    fig.update_annotations(
        font=dict(size=20)
    )  # Change the title size of each subplot


    # -----------Legend----------
    # Add annotations or a custom legend to explain the dash lines:
    # Add the explanation to the legend using a dummy plot
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Negative <br> Control",
            marker=dict(size=12, symbol="line-ew"),
            line=dict(width=4, color="gray", dash="solid"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Parental",
            marker=dict(size=12, symbol="line-ew"),
            line=dict(width=4, color="blue", dash="solid"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Low-uptake",
            marker=dict(size=12, symbol="line-ew"),
            line=dict(width=4, color="green", dash="solid"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="High-uptake",
            marker=dict(size=12, symbol="line-ew"),
            line=dict(width=4, color="red", dash="solid"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="low-uptake <br> thresh",
            marker=dict(size=12, symbol="line-ew"),
            line=dict(width=2, color="green", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="high-uptake <br> thresh",
            marker=dict(size=12, symbol="line-ew"),
            line=dict(width=2, color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text=f"{signal_col} signal intensity", showgrid=False,
        row=len_exp, col=1,
    )

    # fig.update_layout(showlegend=True)
    fig.update_layout(
        font=dict(
            family="Arial"
        ),
        height=1800,
        width=600,
    )
    return fig
