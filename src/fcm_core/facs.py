import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import Image, display


def fcs_time_histogram(
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
        _hover = (
            f"Signal: %{{x:.1f}}<br>"
            f"Count: %{{y}}<br>"
            f"Low threshold: {low_ref:.1f}<br>"
            f"High threshold: {high_ref:.1f}"
            "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=bins,
                y=hist_ctrl,
                name="Control",
                line_color="black",
                fillcolor="rgba(36, 37, 42, 0.5)",
                fill="tozeroy",
                showlegend=False,
                hovertemplate=_hover,
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
                hovertemplate=_hover,
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
                hovertemplate=_hover,
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
                    hovertemplate=_hover,
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


class LinePlot:
    """Collection of Plotly line-plot visualizations for flow cytometry data.

    Common axis styles and layout settings are defined once in ``__init__``
    and reused across all plot methods.

    Args:
        height: Default figure height in pixels.
        width: Default figure width in pixels.
        font_family: Font family applied to the whole figure.
        line_color: Border color for axes.
        line_width: Border width for axes.
        plot_bgcolor: Background color of the plot area.
        tick_font_size: Font size for axis tick labels.
        title_font_size: Font size for axis titles.
    """

    def __init__(
        self,
        height=500,
        width=1200,
        font_family="Arial",
        line_color="black",
        line_width=2,
        plot_bgcolor="white",
        tick_font_size=25,
        title_font_size=25,
    ):
        self.height = height
        self.width = width
        self.font_family = font_family
        self.line_color = line_color
        self.line_width = line_width
        self.plot_bgcolor = plot_bgcolor
        self.tick_font_size = tick_font_size
        self.title_font_size = title_font_size

    def _compute_medians(self, df, signal_col="PKH26", exclude_sample="Dye"):
        """Compute per-experiment, per-sample medians from raw event data.

        Args:
            df: Raw events dataframe. Must have columns ``Experimental_ID``,
                ``Sample``, ``File name``, ``Date_of_experiment``, ``Days``,
                and ``signal_col``.
            signal_col: Fluorescence channel column to compute median on.
                Defaults to ``"PKH26"``.
            exclude_sample: Substring used to drop unwanted samples
                (e.g. ``"Dye"``). Defaults to ``"Dye"``.

        Returns:
            pd.DataFrame: Aggregated medians sorted by ``Experimental_ID``,
            with an additional ``cell_count`` column.
        """
        work_df = df.copy()
        work_df["cell_count"] = 1

        medians = work_df.groupby(
            by=["Experimental_ID", "Sample", "File name",
                "Date_of_experiment", "Days"]
        ).agg({signal_col: "median", "cell_count": "sum"}).reset_index()

        medians = medians.loc[~medians["Sample"].str.contains(exclude_sample)]
        medians.sort_values(by="Experimental_ID", inplace=True)
        return medians

    def _apply_common_styles(self, fig, title_text):
        """Apply shared axis borders, background, and font to a figure."""
        fig.update_xaxes(
            showline=True, linewidth=self.line_width, linecolor=self.line_color,
            mirror=True, ticks="outside",
        )
        fig.update_yaxes(
            showline=True, linewidth=self.line_width, linecolor=self.line_color,
            mirror=True, ticks="outside",
        )
        fig.update_layout(
            plot_bgcolor=self.plot_bgcolor,
            title=dict(text=title_text, x=0.5, font_size=self.title_font_size),
            height=self.height,
            width=self.width,
            font=dict(family=self.font_family),
        )
        return fig

    def fcs_high_low_ratio_lineplot(
        self,
        df,
        signal_col="PKH26",
        low_sample="Low",
        high_sample="High",
        exclude_sample="Dye",
        first_ratio_zero=True,
        on_graph_text=False,
    ):
        """Plot the High-to-Low uptake median ratio as a line graph over time.

        Computes per-experiment medians of ``signal_col`` for each sample
        group, derives the High/Low ratio at each time point, and returns a
        styled Plotly line figure.

        Args:
            df: Normalized dataframe. Must have columns ``Experimental_ID``,
                ``Sample``, ``File name``, ``Date_of_experiment``, ``Days``,
                and ``signal_col``.
            signal_col: Fluorescence channel column for median computation.
                Defaults to ``"PKH26"``.
            low_sample: Label of the low-uptake sample group. Defaults to
                ``"Low"``.
            high_sample: Label of the high-uptake sample group. Defaults to
                ``"High"``.
            exclude_sample: Substring used to filter out unwanted samples
                (e.g. ``"Dye"``). Defaults to ``"Dye"``.
            first_ratio_zero: If ``True``, sets the first time-point ratio
                to zero (representing the initial sort). Defaults to ``True``.
            on_graph_text: If ``True``, annotates each point with its
                ``Date_of_experiment`` label. Defaults to ``False``.

        Returns:
            go.Figure: Plotly line figure with one point per experiment.
        """
        medians = self._compute_medians(
            df, signal_col=signal_col, exclude_sample=exclude_sample
        )

        ratio_label = f"{high_sample}/{low_sample} ratio"
        ratio_col = f"{signal_col}_ratio"

        all_medians = []
        for exp in medians["Experimental_ID"].unique():
            data = medians.loc[medians["Experimental_ID"] == exp]
            row_low = data.loc[data["Sample"] == low_sample]
            row_high = data.loc[data["Sample"] == high_sample]
            result = row_high[
                signal_col
            ].values[0] / row_low[signal_col].values[0]
            new_row = row_high.copy()
            new_row[signal_col] = result
            new_row["Sample"] = ratio_label
            data = pd.concat([data, new_row], ignore_index=True)
            data = data.loc[data["Sample"] == ratio_label]
            data = data.rename(columns={signal_col: ratio_col})
            all_medians.append(data)

        df_ratios = pd.concat(all_medians).reset_index(drop=True)

        if first_ratio_zero:
            df_ratios.loc[0, ratio_col] = 0

        if on_graph_text:
            fig = px.line(
                df_ratios, x="Days", y=ratio_col, markers=True,
                text="Date_of_experiment", hover_data=["cell_count"],
            )
        else:
            fig = px.line(
                df_ratios, x="Days", y=ratio_col, markers=True,
                hover_data=["cell_count"],
            )

        fig.update_xaxes(
            title_text="Days", tick0=1, dtick=1,
            range=[-5, df_ratios["Days"].max() + 5],
            tickfont_size=self.tick_font_size, 
            title_font_size=self.title_font_size,
            tickvals=df_ratios["Days"], showgrid=False,
        )
        fig.update_yaxes(
            title_text="Fold-change",
            tickfont_size=self.tick_font_size, 
            title_font_size=self.title_font_size,
            showgrid=False,
        )
        fig.update_traces(
            line=dict(color="blue", width=5), marker=dict(size=16),
            textfont_size=14,
        )

        title = f"Ratio of {high_sample} to {low_sample} EV-uptake populations over time"
        self._apply_common_styles(fig, title)
        return fig

    def fcs_signals_lineplot(
        self,
        df,
        signal_col="PKH26",
        exclude_sample="Dye",
        x_col="Days",
        color_map=None,
        title="All FACS experiments over time",
    ):
        """Plot per-sample median signal intensity as a line graph over time.

        Args:
            df: Raw events dataframe. Must have columns ``Experimental_ID``,
                ``Sample``, ``File name``, ``Date_of_experiment``, ``Days``,
                and ``signal_col``.
            signal_col: Column name for the fluorescence signal plotted on
                the y-axis. Defaults to ``"PKH26"``.
            exclude_sample: Substring used to drop unwanted samples
                (e.g. ``"Dye"``). Defaults to ``"Dye"``.
            x_col: Column name to use as the x-axis. Defaults to ``"Days"``.
            color_map: Dict mapping sample labels to line colors, e.g.
                ``{"Control": "black", "Low": "blue", "High": "red"}``.
                Defaults to a preset Control/Low/High palette.
            title: Plot title. Defaults to ``"All FACS experiments over time"``.

        Returns:
            go.Figure: Plotly line figure with one trace per sample group.
        """
        medians = self._compute_medians(
            df, signal_col=signal_col, exclude_sample=exclude_sample
        )

        if color_map is None:
            color_map = {"Control": "black", "Low": "blue", "High": "red"}

        fig = px.line(
            medians, x=x_col, y=signal_col, color="Sample",
            color_discrete_map=color_map, markers=True,
        )
        fig.update_xaxes(
            title_text=x_col,
            tickvals=medians[x_col].unique(),
            tickfont_size=self.tick_font_size, 
            title_font_size=self.title_font_size,
        )
        fig.update_yaxes(
            title_text=f"{signal_col} signal intensity",
            tickfont_size=self.tick_font_size, 
            title_font_size=self.title_font_size,
        )

        self._apply_common_styles(fig, title)
        return fig



def ssc_fcs_scatterplot(
    df,
    x_channel="FSC-H",
    y_channel="SSC-H",
    title="FSC vs SSC scatter plot",
    static=True,
    height=2000,
    width=400,
    vertical_spacing=0.07,
    **kwargs
):
    """Plot scatterplot subplots of two channels, one subplot per unique day.

    Creates a vertically stacked figure with one scatterplot per unique value
    in the 'Day' column. Each subplot shows the relationship between the two
    specified channels, colour-coded by sample group.

    Args:
        df: DataFrame containing at minimum a 'Day', 'Sample', and the two
            channel columns.
        x_channel: Column name to use for the x-axis. Defaults to 'FSC-H'.
        y_channel: Column name to use for the y-axis. Defaults to 'SSC-H'.
        title: Overall figure title. Defaults to 'FSC vs SSC scatter plot'.
        static: If True, display the figure statically. Defaults to True.
        height: Height of the figure in pixels. Defaults to 2000.
        width: Width of the figure in pixels. Defaults to 400.
        vertical_spacing: Vertical spacing between subplots (0 to 1). Defaults
            to 0.07.
        kwargs: Additional keyword arguments passed to `fig.to_image()` when
            `static=True`.
    Returns:
        go.Figure: Plotly figure with one subplot row per day.
    """
    color_map = {
        "Control": "black", "Low": "green", "High": "red", "Parental": "blue"
    }

    days = df["Day"].unique()
    n_days = len(days)

    fig = make_subplots(
        rows=n_days,
        cols=1,
        subplot_titles=tuple(days),
        vertical_spacing=vertical_spacing, 
    )

    for i, day in enumerate(days, start=1):
        day_df = df.loc[df["Day"] == day]
        for sample, color in color_map.items():
            sample_df = day_df.loc[day_df["Sample"] == sample]
            if sample_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sample_df[x_channel],
                    y=sample_df[y_channel],
                    mode="markers",
                    name=sample,
                    marker=dict(color=color, size=3, opacity=0.5),
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

        fig.update_xaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_text=x_channel,
            title_font_size=16,
            tickfont_size=14,
            showgrid=False,
            row=i,
            col=1,
        )
        fig.update_yaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            title_text=y_channel,
            title_font_size=16,
            tickfont_size=14,
            showgrid=False,
            row=i,
            col=1,
        )

    fig.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=20,
        height=height,
        width=width,
        plot_bgcolor="white",
    )
    fig.update_annotations(font=dict(size=18))

    if static:
        img_bytes = fig.to_image(
            format="png", 
            height=kwargs.get("height", height),
            width=kwargs.get("width", width)
        )
        display(Image(data=img_bytes))
        return None
    else:
        return fig


