import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List
from pathlib import Path
from FlowCytometryTools import FCMeasurement

#%%
def compile_fcs_data(
    data_folder="data", 
    sample_name_map=None, 
    incubation_map=None,
    include_keywords=None,
    exclude_keywords=None
):
    """
    Compiles FCS experiment data from subfolders into a single pandas 
    DataFrame.
    
    Args:
    - data_folder (str): Name of the folder containing the data.
    - sample_name_map (dict): {substring: "Sample Name"}
    - incubation_map (dict): {substring: "Duration Label"}
    - include_keywords (list): Only process files containing AT LEAST ONE 
        of these strings in the name.
    - exclude_keywords (list): Skip files containing ANY of these strings in 
        the name.

    Returns:
        pd.DataFrame: A compiled DataFrame with metadata and FCS data.
    """
   
    # Standardize inputs: Convert single strings to lists
    if isinstance(include_keywords, str):
        include_keywords = [include_keywords]
    if isinstance(exclude_keywords, str):
        exclude_keywords = [exclude_keywords]

    # Initialize defaults
    sample_name_map = sample_name_map or {}
    incubation_map = incubation_map or {}
    include_keywords = include_keywords or []
    exclude_keywords = exclude_keywords or []

    # Define the base path
    folders_path = Path(f"./../{data_folder}")
    print(f"Looking for data in: {folders_path.resolve()}")
    print("="*30 + "\n")
    
    if not folders_path.exists():
        raise FileNotFoundError(
            f"The directory {folders_path.resolve()} does not exist."
            )

    # If we also use an inclusion criteria filter, the included 
    # files will be added to this list.
    dfs_list = [] 
    excluded_files = [] 

     # Make a section heading that prints that outcome of the following loop.
    print("Processed files")
    print("-" * 30)
    # rglob searches recursively through all nested folders for .fcs files
    for file_path in folders_path.rglob("*.fcs"):
       
        file_name = file_path.name
     
        # --- FILTERING LOGIC ---
        # First we check the exclusion criteria, because we can skip that file 
        # if it meets any of the exclusion criteria.      
        if exclude_keywords:
            if any(kw in file_name for kw in exclude_keywords):
                excluded_files.append(file_name)
                continue
    
        # Then we check the inclusion criteria. If exclusion logic comes after 
        # the inclusion logic, we might end up processing files that should have
        # been excluded but they are not, just because they met the inclusion 
        # criteria.

        if include_keywords:
            # If include_keywords are provided, only process files that contain 
            # at least one of the keywords
            if not any(kw in file_name for kw in include_keywords):
                continue # Skip this file and move to the next one
                # If we reach this point, it means the file matches the 
                # inclusion criteria
                
       
        # Load FCS data
        sample = FCMeasurement(ID='Test Sample', datafile=str(file_path))
        df_partial = sample.data.copy() 
        
        # 1. Determine Sample Name 
        sample_name = "Unknown" 
        for key, name in sample_name_map.items():
            if key in file_name:
                sample_name = name
                break # Stop at the first match found

        # 2. Determine Incubation Duration
        incub_duration = "Unknown"
        for key, duration in incubation_map.items():
            if key in file_name:
                incub_duration = duration
                break

        # 3. Extract Date/Experiment ID from the immediate parent folder
        folder_name = file_path.parent.name

        # Insert metadata into the first few columns
        df_partial.insert(0, "Sample", sample_name)
        df_partial.insert(1, "Incubation_duration", incub_duration)
        df_partial.insert(2, "Date", folder_name)
        df_partial.insert(3, "File name", file_name)
        
        dfs_list.append(df_partial)

        print(f"File name: {file_name}")

    print("=" * 30 + "\n") # at the end of the loop, we print a summary of 
                           # processed files.

    if include_keywords and not dfs_list:
        raise ValueError(
            "Inclusion criteria was not met: No files found containing " \
            "the provided include keywords."
        )
    # But if we have an inclusion criteria and we found files that met that 
    # criteria, we print a summary of the included files. We don't want to use
    # `else` because we may have an inclusion criteria but still have
    # dfs_list, and therefore, we do not want to print a message in that case.   
    elif include_keywords and dfs_list:
        print("An Inclusion criteria was set")
        print("-" * 30)
        print(f"The number of included .fcs files: {len(dfs_list)}")
        print("=" * 30 + "\n")
        
    if exclude_keywords:
        print("Excluded files")
        print("-" * 30)
        if excluded_files:
            print(f"Excluded the following files: {', '.join(excluded_files)}")
        else:
            print("No files were excluded.")
        print("=" * 30 + "\n")
    # Combine all individual dataframes into one large master dataframe
    if not dfs_list:
        print(f"No matching .fcs files found in {folders_path.resolve()}")
        return pd.DataFrame() 
    
    # combine all the dataframes in the list into one master dataframe
    df = pd.concat(dfs_list, ignore_index=True)

    # --- Integrated Summary Printing ---
    print("FCS COMPILATION SUMMARY")
    print("-"*30)
    samples_per_file = (
        df.groupby(["File name", "Sample"])
          .size()
          .reset_index(name="Count (Events)")
          .sort_values(["File name", "Sample"])
    )
    print(samples_per_file.to_string(index=False))
    print("="*30 + "\n")

    # ----- groupby summary based on sample -------
    print("TOTAL EVENTS PER SAMPLE")
    print("-" * 30)
    
    sample_summary = (
        df.groupby("Sample")
          .size()
          .reset_index(name="Total count (Events)")
          .sort_values("Total count (Events)", ascending=False)
    )
    
    print(sample_summary.to_string(index=False))
    print("=" * 30 + "\n")

    return df

#%%
def sample_size_normalizer(df, samples_col:str, target_size:int, col:str):
    """ Normalize the sample size of each sample group of flow cytometry 
    (e.g., compound, control, ...) based on a specific sample size. This shoudl
    be done before histogram generation. This method maintains the shape and 
    relative distance between points compared to the other normalization 
    methods that can distort the look of smaller sub-populations.

    Args:
        df: Dataframe containig one or more samples.
        sample_col: The column containig the name of the sample(s)
        target_size: The number we want to normalize different samples based 
                     on (For example, 30000 cells).
        col: column with the value that should be adjusted based on the 
             normalization factor for sample size.
    Returs:
        A dataframe with the same number of rows as the original one, but with 
        a new column that is the normalized value of the given column based on 
        the sample size normalization factor.
    """
    import pandas as pd

    final_list = []
    for sample in df[samples_col].unique():
        # Select the sample
        sample_df = df.loc[df[samples_col] == sample]
        # Get the number of rows in the sample df, which is the sample size. 
        sample_df_length = len(sample_df) 
        # Make a copy of the sample df to avoid modifying the original one.
        sample_df = sample_df.copy() 
        # Apply the normalization factor to the values of the given column and 
        # create a new column with the normalized values.
        sample_df["Size_normalized" + "_" + col] = sample_df[col].apply(
            lambda x: (target_size/sample_df_length) * x
            ) 
        # Append the modified sample df to the final list.
        final_list.append(sample_df) 
    final_df = pd.concat(final_list) # Concatenate all the modified sample dfs 
                                     # in the final list to get the final df.
    return final_df
        

#%%
class Graph:
    """
    A class to generate and customize Flow Cytometry (FCM) histograms.

    Attributes:
        bins (np.array): The bin edges for the histogram.
        width (int): Width of the plot in pixels.
        height (int): Height of the plot in pixels.
        plot_bgcolor (str): Background color of the plot.
        x_title (str): Title for the x-axis.
        y_title (str): Title for the y-axis.
        title_font_size (int): Font size for the plot title.
        tick_font_size (int): Font size for the axis ticks.
        border_width (int): Width of the plot borders.
        border_color (str): Color of the plot borders.
        show_line (bool): Whether to show the axis lines.
        mirror_axes (bool): Whether to mirror the axis lines on all sides.
        ticks_style (str): Style of the axis ticks (e.g., 'outside', 'inside', 'none').
        tick0 (int): Starting point for the y-axis ticks.
        dtick (int): Step size for the y-axis ticks.
        minor_dtick (int): Step size for the minor ticks on the y-axis.
        line_colors (List[str]): List of colors for the histogram lines.
        fill_colors (List[str]): List of colors for the histogram fills.
    """
 
    def __init__(self, 
                 df, 
                 samples= List[str], 
                 incub_duration=None, 
                 date=None,
                 channel="Size_normalized_PKH26"):
        
        """
        Args:
            df (pd.DataFrame): The input dataframe containing FCM data.
            samples (List[str]): List of sample names to include in the graph.
            incub_duration (str, optional): Incubation duration to filter 
                the data.
            date (str, optional): Date/Experiment ID to filter the data.
            channel (str): The column name in the dataframe to be plotted on 
                the x-axis.

        Raises:
            ValueError: If the 'samples' list is empty, or if any of the 
                provided 'samples', 'incub_duration', or 'date' values are not 
                present in the dataframe's respective columns.
        """

        # Check the samples are defined
        if not samples: 
            raise ValueError("The 'samples' list cannot be empty. " \
            "Please provide at least one sample name.")
        
        
        # Check if the input for the parameters `samples` and `incub_duration`
        # exist in the dataframe.
        if not all(sample in df["Sample"].unique() for sample in samples):
            missing_samples = [
                sample for sample in samples 
                if sample not in df["Sample"].unique()
                ]
            raise ValueError(
                f"The following samples are not found in the dataframe:"
                f" {', '.join(missing_samples)}"
                ) 

        
        if (incub_duration is not None 
                and incub_duration not in df["Incubation_duration"].unique()):
            raise ValueError(
                f"The incubation duration '{incub_duration}' is not found"
                 "in the dataframe."
                )

        if date is not None and date not in df["Date"].unique():
            raise ValueError(
                f"The date '{date}' is not found in the dataframe."
                )

        # --- Data Attributes ---
        self.df = df
        self.samples = samples
        self.incub_duration = incub_duration
        self.channel = channel
        # Note: As the packages are imported inside the class, we need to use 
        # self.np to access numpy in this code.
        self.bins = self.np.logspace(0, 6, 256) 
        self.date = date

        # --- Layout Attributes ---
        # Dimensions and Background
        self.width = 600
        self.height = 500
        self.plot_bgcolor = "white"
        
        # --- Titles and Fonts ---
        self.x_title = "PKH26 signal intensity"
        self.y_title = "Count"
        self.title_font_size = 20
        self.tick_font_size = 20
        
        # Borders and Lines
        self.border_width = 2
        self.border_color = 'black'
        self.show_line = True
        self.mirror_axes = True
        self.ticks_style = 'outside'

        # --- Y-Axis Ticks ---
        # The following parameters for placing ticks on the y-axis was kept 
        # flexible as if the range of counts on y-axis is too different, plotly 
        # will hide the numbers. In those cases, we need to change these to 
        # None to let plotly decides the best values.
        self.tick0 = 0
        self.dtick = 50
        self.minor_dtick = 10 # the distance between minor ticks on the y-axis 
                              # (x-axis is always the same log scale and
                              # doees not need to be changed).

        # --- Color Palettes ---
        self.line_colors = [
            'black', 'blue', 'red', 'green', 
            'purple', 'orange', 'cyan'
        ]
        self.fill_colors = [
            "rgba(36, 37, 42, 0.5)",     # Blackish
            "rgba(45, 85, 255, 0.5)",    # Blueish
            "rgba(214, 39, 40, 0.5)",    # Reddish
            "rgba(44, 160, 44, 0.5)",    # Greenish
            "rgba(148, 103, 189, 0.5)",  # Purplish
            "rgba(255, 127, 14, 0.5)",   # Orangeish
            "rgba(23, 190, 207, 0.5)"    # Cyanish
        ]

    def set_layout(self, **kwargs):
        """
        Utility method to quickly update multiple layout attributes at once.
       
        Args:
            **kwargs: Any layout attribute of the Graph class that you want to 
                      update. For example, width, height, plot_bgcolor, etc.
        Returns:
            None

        Example: graph.set_layout(
                    width=800, 
                    height=600, 
                    plot_bgcolor="lightgray"
                 )
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: '{key}' is not a recognized attribute.")

    def graph_generator(self, save_fig=False, save_path: str = None):
        """
        Filters the dataframe, builds the Plotly figures based on current 
        attributes, and displays/saves them.
        
        Args:
            save_fig (bool): Whether to save the figure as an SVG file.
            save_path (str): The directory path where the figure should be 
                saved. If None, the figure will not be saved.
        Returns:
            None
        """

        selected_samples = "|".join(self.samples)
        selected_experiments = self.df.loc[
            (self.df["Sample"].str.contains(selected_samples)) & 
            ((self.df["Incubation_duration"] == self.incub_duration)
              if self.incub_duration is not None else True) &
            ((self.df["Date"] == self.date) 
              if self.date is not None else True)
        ]
        
        for exp in selected_experiments["Date"].unique():
            fig = self.go.Figure()
            exp_df = selected_experiments.loc[
                        selected_experiments["Date"] == exp
            ]
            # Add a trace for each sample dynamically
            for idx, sample_name in enumerate(self.samples):
                sample_data = exp_df[self.channel].loc[
                    exp_df["Sample"] == sample_name
                ]

                if sample_data.empty:
                    continue
                
                hist_data, _ = self.np.histogram(sample_data, bins=self.bins)
                l_color = self.line_colors[idx % len(self.line_colors)]
                f_color = self.fill_colors[idx % len(self.fill_colors)]
                
                fig.add_trace(
                    self.go.Scatter(
                        x=self.bins, y=hist_data, name=sample_name, 
                        line_color=l_color, line_width=1,
                        fillcolor=f_color, fill='tozeroy', opacity=0.5
                    )   
                ) 

            # X-Axis settings
            fig.update_xaxes(
                type='log',
                exponentformat='power', 
                showline=self.show_line, 
                linewidth=self.border_width, 
                linecolor=self.border_color, 
                mirror=self.mirror_axes,
                ticks=self.ticks_style,
                tick0=0, dtick=1, # we keep this default as the x-axis range is
                                  # fixed across all experiments.        
                minor=dict(
                    ticks=self.ticks_style, 
                    showgrid=False, 
                    dtick='D1'
                ),
            
                title_text=self.x_title, 
                title_font_size=self.title_font_size, 
                tickfont_size=self.tick_font_size
            )
            
            # Y-Axis settings
            fig.update_yaxes(
                showline=self.show_line, 
                linewidth=self.border_width, 
                linecolor=self.border_color, 
                mirror=self.mirror_axes,
                ticks=self.ticks_style,
                # The ticks are made flexible only for the y-axis, as it can 
                # be very varibale.   
                tick0=self.tick0, 
                dtick=self.dtick,  
                minor=dict(
                    ticks=self.ticks_style, 
                    showgrid=False, 
                    dtick=self.minor_dtick
                ), 
                title_text=self.y_title, 
                title_font_size=self.title_font_size, 
                tickfont_size=self.tick_font_size
            )

            # Global Layout settings
            duration = exp_df["Incubation_duration"].unique()[0]
            title = f"Sorting {exp}<br>{duration}"
            fig.update_layout(
                title=title, 
                title_x=0.5,
                height=self.height, 
                width=self.width, 
                plot_bgcolor=self.plot_bgcolor
            )

            # Handle Export
            if save_fig:
                if save_path is not None:
                    exp_name = f"Sorting {exp}_{duration}"
                    file_name = f"/FACS hist_ {exp_name}"
                    fig.write_image(f"{save_path}{file_name}.svg")
                else:
                    print(
                        "Warning: save_path is not provided. " \
                        "Figure will not be saved."
                    )

            fig.show()