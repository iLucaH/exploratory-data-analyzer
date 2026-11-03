import argparse
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_DATASET_PATH = "example_data.csv"
DEFAULT_OUTPUT_NAMESPACE = "data_explore_export"
# DEFAULT_PLOT_TYPE_CHOICES is built from the PLOT_FUNCTIONS registry further down.
# A hue column with more unique values than this is treated as continuous.
MAX_CATEGORICAL_HUE_VALUES = 12

verbose = False


# Basic logging for verbose outputs. Enabled through the command line argument --verbose.
def log(text):
    if verbose:
        print('\n')
        print(text)


def load_dataset(path):
    """
    Loads a CSV dataset into a pandas DataFrame and prints basic
    information about the dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        The file path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the dataset if the file is loaded
        successfully, or None if the file cannot be found.
    """
    log("Loading in a new dataset at path: " + str(path) + '.')

    try:
        pandas_dataframe = pd.read_csv(path)
        log('Successfully read in the dataset at path: ' + str(path) + '.')
    except FileNotFoundError:
        print('ERROR: The path: ' + str(path) + ' could not be found!')
        return None

    log('Printing dataset column headings')
    if verbose:
        for column in pandas_dataframe.columns:
            print(column)
    log('Printing the shape of the dataset')
    if verbose:
        print(pandas_dataframe.shape)
    log('Printing the data types in the dataset')
    if verbose:
        print(pandas_dataframe.dtypes)
    log('Printing information about the dataset')
    if verbose:
        pandas_dataframe.info()  # info() already prints. No need to wrap it in print.

    return pandas_dataframe


def clean_dataset(pandas_dataframe):
    """
    Cleans a pandas DataFrame by removing duplicate rows, converting
    columns to numerical values, and removing rows containing missing
    or invalid values.

    Parameters
    ----------
    pandas_dataframe : pandas.DataFrame
        The DataFrame containing the dataset to clean.

    Returns
    -------
    pandas.DataFrame or None
        A cleaned DataFrame, or None if nothing usable remains.
    """
    # Remove duplicates first so later operations don't see duplicate values.
    previous_dataset_length = len(pandas_dataframe)
    log('Removing duplicate rows from the dataset where the original length is: ' + str(previous_dataset_length))
    pandas_dataframe = pandas_dataframe.drop_duplicates().copy()
    num_removed = previous_dataset_length - len(pandas_dataframe)
    log('New length: ' + str(len(pandas_dataframe)) + '. Removed: ' + str(num_removed) + ' rows.')

    # Non-numerical values become NaN because of errors='coerce'.
    log('Transforming the dataframe into numerical columns.')
    for column in pandas_dataframe.columns:
        pandas_dataframe[column] = pd.to_numeric(pandas_dataframe[column], errors='coerce')

    log('Numerical transform result:')
    log(pandas_dataframe.dtypes)
    log('Checking the number of total values.')
    log(pandas_dataframe.count())
    log('Checking the number of not available values.')
    log(pandas_dataframe.isna().sum())

    # Remove rows containing missing or invalid numerical values.
    log('Removing rows with invalid numbers.')
    previous_dataset_length = len(pandas_dataframe)
    pandas_dataframe = pandas_dataframe.dropna()
    num_removed = previous_dataset_length - len(pandas_dataframe)
    log('New length: ' + str(len(pandas_dataframe)) + '. Removed: ' + str(num_removed) + ' rows.')

    if pandas_dataframe.empty:
        return None
    return pandas_dataframe


def validate_columns(pandas_dataframe, columns):
    """Returns the requested column names that do not exist in the DataFrame."""
    return [column for column in columns if column not in pandas_dataframe.columns]


def prepare_hue(pandas_dataframe, hue):
    """
    Returns a copy of the DataFrame where a low-cardinality hue column is
    converted to an ordered categorical, so plots use discrete colours.
    Columns with many unique values are left numeric (continuous).
    """
    if hue is None:
        return pandas_dataframe
    data = pandas_dataframe.copy()
    if data[hue].nunique() <= MAX_CATEGORICAL_HUE_VALUES:
        data[hue] = pd.Categorical(data[hue])
    return data


def perform_numerical_analysis(pandas_dataframe, columns):
    """
    Calculates and prints the mean, median, and standard deviation
    of each of the selected columns.

    Parameters
    ----------
    pandas_dataframe : pandas.DataFrame
        The DataFrame containing the dataset to analyse.
    columns : list[str]
        The numerical columns to summarise.

    Returns
    -------
    pandas.DataFrame
        A summary DataFrame (also printed) with one column per selected
        column and rows for Mean, Median and Std.
    """
    log('Performing numerical analysis on the following columns: ' + ', '.join(columns))

    summary = pd.DataFrame({
        column: [
            np.mean(pandas_dataframe[column]),    # Mean
            np.median(pandas_dataframe[column]),  # Median
            np.std(pandas_dataframe[column])      # Standard deviation
        ]
        for column in columns
    }, index=['Mean', 'Median', 'Std'])

    print(summary)
    return summary


def plot_histogram(pandas_dataframe, columns, hue=None):
    """
    Creates a grid of histograms, one per selected column. If a hue
    column is given, each histogram is split by the hue values.

    Parameters
    ----------
    pandas_dataframe : pandas.DataFrame
        The DataFrame containing the dataset to analyse.
    columns : list[str]
        The columns to draw a distribution for.
    hue : str, optional
        A column used to split each histogram into groups.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the histograms.
    """
    log('Creating a histogram of the data.')
    data = prepare_hue(pandas_dataframe, hue)

    n_plots = len(columns)
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for ax, column in zip(axes.flat, columns):
        sns.histplot(data=data, x=column, hue=hue, bins=15, edgecolor="black", ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    # Hide any unused grid cells.
    for ax in list(axes.flat)[n_plots:]:
        ax.set_visible(False)

    title = "Distributions of " + ", ".join(columns)
    if hue:
        title += f" by {hue}"
    figure.suptitle(title)
    figure.tight_layout()

    return figure


def multi_variable_scatterplot(pandas_dataframe, columns, hue=None):
    """
    Creates a scatterplot of the first two selected columns. If a hue
    column is given, points are coloured by it and a colourbar is added.

    Parameters
    ----------
    pandas_dataframe : pandas.DataFrame
        The DataFrame containing the dataset to analyse.
    columns : list[str]
        At least two columns. The first is the x axis, the second the y axis.
    hue : str, optional
        A column used to colour the points.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the scatterplot.
    """
    log('Creating a scatterplot of the data.')
    x_column, y_column = columns[0], columns[1]

    figure, ax = plt.subplots(figsize=(9, 6))

    scatter_kwargs = dict(s=25, alpha=0.65, edgecolors="none")
    if hue:
        scatter_plot = ax.scatter(
            pandas_dataframe[x_column],
            pandas_dataframe[y_column],
            c=pandas_dataframe[hue],
            cmap="viridis",
            **scatter_kwargs
        )
        colourbar = figure.colorbar(scatter_plot, ax=ax)
        colourbar.set_label(hue, fontsize=11)
        title = f"{x_column} vs {y_column} by {hue}"
    else:
        ax.scatter(pandas_dataframe[x_column], pandas_dataframe[y_column], **scatter_kwargs)
        title = f"{x_column} vs {y_column}"

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(x_column, fontsize=11)
    ax.set_ylabel(y_column, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.25)
    figure.tight_layout()

    return figure


def seaborn_pair_plot(pandas_dataframe, columns, hue=None):
    """
    Creates a Seaborn pairplot showing the relationships between the
    selected columns, optionally coloured by a hue column.

    Parameters
    ----------
    pandas_dataframe : pandas.DataFrame
        The DataFrame containing the dataset.
    columns : list[str]
        At least two columns to compare.
    hue : str, optional
        A column used to colour the points.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the pairplot.
    """
    log("Creating a pairplot of the data")

    # The hue column may or may not already be one of the plotted columns.
    plot_columns = list(columns)
    if hue and hue not in plot_columns:
        plot_columns.append(hue)
    plot_data = prepare_hue(pandas_dataframe[plot_columns], hue)

    pairplot = sns.pairplot(
        plot_data,
        vars=columns,
        hue=hue,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.5, "s": 15},
        diag_kws={"bins": 15}
    )
    title = "Relationships Between " + ", ".join(columns)
    if hue:
        title += f" by {hue}"
    pairplot.figure.suptitle(title, y=1.02)

    return pairplot.figure


def make_grid(n_plots, width=6, height=4):
    """Creates a roughly square grid of subplots with n_plots visible axes."""
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows), squeeze=False)
    for ax in list(axes.flat)[n_plots:]:
        ax.set_visible(False)
    return figure, list(axes.flat)[:n_plots]


def title_suffix(columns, hue):
    """Builds the 'col1, col2 by hue' part of a figure title."""
    title = ", ".join(columns)
    if hue:
        title += f" by {hue}"
    return title


# ---------------------------------------------------------------------------
# Additional plots.
# Every plot has the same signature: (pandas_dataframe, columns, hue=None) -> Figure
# To add a new one, write a function like these and add it to PLOT_FUNCTIONS
# (and to PLOT_MIN_COLUMNS if it needs more than one column).
# ---------------------------------------------------------------------------

def plot_boxplot(pandas_dataframe, columns, hue=None):
    """
    One box plot per selected column. With a hue column, there is one
    box per hue group; otherwise a single box per column.
    """
    log('Creating box plots of the data.')
    data = prepare_hue(pandas_dataframe, hue)
    figure, axes = make_grid(len(columns))

    for ax, column in zip(axes, columns):
        if hue:
            sns.boxplot(data=data, x=hue, y=column, ax=ax)
        else:
            sns.boxplot(data=data, y=column, ax=ax)
        ax.set_title(f"Box plot of {column}")

    figure.suptitle("Box plots of " + title_suffix(columns, hue))
    figure.tight_layout()
    return figure


def plot_violinplot(pandas_dataframe, columns, hue=None):
    """
    One violin plot per selected column, showing the full shape of the
    distribution. With a hue column, there is one violin per hue group.
    """
    log('Creating violin plots of the data.')
    data = prepare_hue(pandas_dataframe, hue)
    figure, axes = make_grid(len(columns))

    for ax, column in zip(axes, columns):
        if hue:
            sns.violinplot(data=data, x=hue, y=column, inner="quartile", cut=0, ax=ax)
        else:
            sns.violinplot(data=data, y=column, inner="quartile", cut=0, ax=ax)
        ax.set_title(f"Violin plot of {column}")

    figure.suptitle("Violin plots of " + title_suffix(columns, hue))
    figure.tight_layout()
    return figure


def plot_heatmap(pandas_dataframe, columns, hue=None):
    """
    A correlation heatmap between the selected columns (Pearson). The
    hue column is ignored here. Needs at least two columns.
    """
    log('Creating a correlation heatmap of the data.')
    correlations = pandas_dataframe[columns].corr()

    size = max(6, 1.1 * len(columns))
    figure, ax = plt.subplots(figsize=(size + 1, size))
    mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)  # Hide the duplicate upper triangle.
    sns.heatmap(
        correlations, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Between " + ", ".join(columns))
    figure.tight_layout()
    return figure


def plot_regression(pandas_dataframe, columns, hue=None):
    """
    Scatter plot of the first two columns with a fitted regression line
    and the Pearson correlation in the title. If a hue column is given,
    the points are coloured by it while the line is fitted to all points.
    """
    log('Creating a regression plot of the data.')
    x_column, y_column = columns[0], columns[1]
    data = prepare_hue(pandas_dataframe, hue)
    correlation = np.corrcoef(data[x_column], data[y_column])[0, 1]

    figure, ax = plt.subplots(figsize=(9, 6))
    if hue:
        sns.scatterplot(data=data, x=x_column, y=y_column, hue=hue, alpha=0.6, s=25, ax=ax)
        sns.regplot(data=data, x=x_column, y=y_column, scatter=False, color="black", ax=ax)
    else:
        sns.regplot(data=data, x=x_column, y=y_column, scatter_kws={"alpha": 0.5, "s": 25}, ax=ax)

    ax.set_title(f"{x_column} vs {y_column} (r = {correlation:.2f})")
    ax.grid(True, linestyle="--", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_kde(pandas_dataframe, columns, hue=None):
    """
    A kernel density estimate per selected column, a smoothed version of
    the histogram. With a hue column, one filled curve per hue group.
    """
    log('Creating KDE plots of the data.')
    data = prepare_hue(pandas_dataframe, hue)
    figure, axes = make_grid(len(columns))

    for ax, column in zip(axes, columns):
        sns.kdeplot(data=data, x=column, hue=hue, fill=True, common_norm=False, alpha=0.4, ax=ax)
        ax.set_title(f"Density of {column}")

    figure.suptitle("Density plots of " + title_suffix(columns, hue))
    figure.tight_layout()
    return figure


def plot_ecdf(pandas_dataframe, columns, hue=None):
    """
    An empirical cumulative distribution plot per selected column: the
    proportion of rows at or below each value. Good for comparing groups
    without choosing bin sizes.
    """
    log('Creating ECDF plots of the data.')
    data = prepare_hue(pandas_dataframe, hue)
    figure, axes = make_grid(len(columns))

    for ax, column in zip(axes, columns):
        sns.ecdfplot(data=data, x=column, hue=hue, ax=ax)
        ax.set_title(f"Cumulative distribution of {column}")

    figure.suptitle("ECDF plots of " + title_suffix(columns, hue))
    figure.tight_layout()
    return figure


def plot_jointplot(pandas_dataframe, columns, hue=None):
    """
    A scatter plot of the first two columns with the distribution of each
    shown along the margins. Optionally coloured by the hue column.
    """
    log('Creating a joint plot of the data.')
    x_column, y_column = columns[0], columns[1]
    data = prepare_hue(pandas_dataframe, hue)

    joint = sns.jointplot(
        data=data, x=x_column, y=y_column, hue=hue, kind="scatter",
        height=7, joint_kws={"alpha": 0.6, "s": 25}
    )
    joint.figure.suptitle(f"{x_column} vs {y_column}" + (f" by {hue}" if hue else ""), y=1.02)
    return joint.figure


def plot_barplot(pandas_dataframe, columns, hue=None):
    """
    Bar plots of the mean of each selected column. With a hue column,
    one bar per hue group (with a confidence interval); otherwise one bar
    per column showing the mean.
    """
    log('Creating bar plots of the data.')
    if not hue:
        means = pandas_dataframe[columns].mean()
        figure, ax = plt.subplots(figsize=(max(6, 1.5 * len(columns)), 5))
        sns.barplot(x=means.index, y=means.values, ax=ax)
        ax.set_title("Mean of " + ", ".join(columns))
        ax.set_ylabel("Mean")
        figure.tight_layout()
        return figure

    data = prepare_hue(pandas_dataframe, hue)
    figure, axes = make_grid(len(columns))
    for ax, column in zip(axes, columns):
        sns.barplot(data=data, x=hue, y=column, ax=ax)
        ax.set_title(f"Mean {column} by {hue}")

    figure.suptitle("Mean of " + title_suffix(columns, hue))
    figure.tight_layout()
    return figure


# Registry of every available plot. The command line choices, dispatch and
# export file names are all driven by this, so a new plot only needs an entry here.
PLOT_FUNCTIONS = {
    'histogram': plot_histogram,
    'scatterplot': multi_variable_scatterplot,
    'pairplot': seaborn_pair_plot,
    'boxplot': plot_boxplot,
    'violinplot': plot_violinplot,
    'heatmap': plot_heatmap,
    'regression': plot_regression,
    'kde': plot_kde,
    'ecdf': plot_ecdf,
    'jointplot': plot_jointplot,
    'barplot': plot_barplot,
}

# Minimum number of -c columns each plot needs (default is 1).
PLOT_MIN_COLUMNS = {
    'scatterplot': 2,
    'pairplot': 2,
    'heatmap': 2,
    'regression': 2,
    'jointplot': 2,
}

DEFAULT_PLOT_TYPE_CHOICES = list(PLOT_FUNCTIONS)


def export_figure(figure, output_namespace, suffix):
    """Saves a figure as <output_namespace>_<suffix>.png and returns the file name."""
    file_name = f"{output_namespace}_{suffix}.png"
    log('Exporting the ' + suffix + ' to file ' + file_name + '...')
    figure.savefig(file_name, dpi=300, bbox_inches="tight")
    return file_name


def main():
    parser = argparse.ArgumentParser(
        description="A simple project using the Python Standard Library, NumPy, Matplotlib Pyplot, Pandas and Seaborn to analyze numerical datasets quickly and easily.",
        usage="python3 lucas_hautrive.py -p <path> -c <columns...> [-pt <plot_type>] [--hue <column>] [-na] [-v] [-e] [-o <output_namespace>]",
        epilog='Example: python3 lucas_hautrive.py -p data.csv -c alcohol pH -pt pairplot --hue quality -na -v -e -o 47611472'
    )

    parser.add_argument('-p', '--path', type=pathlib.Path, default=DEFAULT_DATASET_PATH, help="File path to the dataset.")
    parser.add_argument('-pt', '--plottype', type=str, default=None, help="Specify a type of plot to create.", choices=DEFAULT_PLOT_TYPE_CHOICES)
    parser.add_argument('-c', '--columns', nargs='+', help="Specify columns to operate on.")
    parser.add_argument('--hue', type=str, default=None, help="Column used to colour/group the plots (e.g. a quality score).")
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_NAMESPACE, help="The namespace used to name the output files.")
    parser.add_argument('-na', '--numerical-analysis', action='store_true', help="Perform a numerical analysis.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose outputs enabled. To visualise charts/diagrams/plots, this flag must be set.")
    parser.add_argument('-e', '--export', action='store_true', help="Auto export charts/diagrams/plots.")

    args = parser.parse_args()
    path = args.path
    plot_type = args.plottype
    columns = args.columns
    hue = args.hue
    is_performing_numerical_analysis = args.numerical_analysis
    global verbose
    verbose = args.verbose
    is_exporting = args.export
    output_namespace = args.output

    # Load the file
    pandas_dataframe = load_dataset(path)
    if pandas_dataframe is None:
        log('A pandas dataframe couldnt be initialised. The program is stopping.')
        return

    # Data cleaning
    pandas_dataframe = clean_dataset(pandas_dataframe)
    if pandas_dataframe is None:
        print('Failed to clean the dataset. Is your dataset numerical? The program is stopping.')
        return

    if not columns:
        print("No columns were provided. Skipping any proceeding operations.")
        return
    print("Proceeding operations will use the following columns: ", columns)

    # Validate every column that was named, including the hue column.
    requested = columns + ([hue] if hue else [])
    missing = validate_columns(pandas_dataframe, requested)
    if missing:
        print("ERROR: These columns are not in the dataset: " + ", ".join(missing))
        print("Available columns: " + ", ".join(pandas_dataframe.columns))
        return

    # Numerical analysis
    if is_performing_numerical_analysis:
        perform_numerical_analysis(pandas_dataframe, columns)
    else:
        log('Numerical analysis flag not set. Skipping any numerical analysis.')

    # Plot using the plot type specified
    figure = None
    plot_function = PLOT_FUNCTIONS.get(plot_type)
    if plot_function is not None:
        min_columns = PLOT_MIN_COLUMNS.get(plot_type, 1)
        if len(columns) < min_columns:
            print(f"ERROR: A {plot_type} needs at least {min_columns} columns.")
            return
        figure = plot_function(pandas_dataframe, columns, hue)
    else:
        log('No plot type specified. Skipping any plotting.')

    if figure is not None:
        if is_exporting:
            file_name = export_figure(figure, output_namespace, plot_type)
            print("Saved " + file_name)
        if verbose:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()