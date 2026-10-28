import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

DEFAULT_DATASET = "winequality_data.csv"
DEFAULT_NUMERICAL_ANALYSIS_COLUMNS = ['pH', 'alcohol']
DEFAULT_OUTPUT_NAMESPACE = "lucas_hautrive_47611472_"

verbose = False

def log(text: str):
    if verbose:
        print('\n')
        print(text)

def load_dataset(path):
    log("Loading in a new dataset at path: " + path + '.')
    
    pandas_dataframe = None
    try:
        pandas_dataframe = pd.read_csv(path)
        log('Successfully read in the dataset at path: ' + path + '.')
    except FileNotFoundError:
        log('ERROR: The path: ' + path + ' could not be found!')
        return
    log('Printing dataset column headings')
    for column in pandas_dataframe.columns:
        print(column)
    log('Printing the shape of the dataset')
    print(pandas_dataframe.shape)
    log('Printing the data types in the dataset')
    print(pandas_dataframe.dtypes)
    log('Printing information about the dataset')
    print(pandas_dataframe.info())
    
    return pandas_dataframe
    
def clean_dataset(pandas_dataframe):
    # Remove any duplicate rows first so any operations afterwards dont operate on
    # duplicate values.
    previous_dataset_length = len(pandas_dataframe)
    log('Removing duplicate rows from the dataset where the original length is: ' + str(previous_dataset_length))
    pandas_dataframe = pandas_dataframe.drop_duplicates()
    num_removed = previous_dataset_length - len(pandas_dataframe)
    log('New length: ' + str(len(pandas_dataframe)) + '. Removed: ' + str(num_removed) + ' rows.')
    
    # Transforming the data into legible numerical floating point values. Any values
    # that are not numerical by default becomes a NaN because of the errors='coerce' option.
    log('Transforming the dataframe into numerical columns.')
    for column in pandas_dataframe.columns:
        pandas_dataframe[column] = pd.to_numeric(pandas_dataframe[column], errors='coerce')
        
    log('Numerical transform result:')
    log(pandas_dataframe.dtypes)
    
    log('Checking the number of total values.')
    log(pandas_dataframe.count())
    log('Checking the number of not available values.')
    log(pandas_dataframe.isna().sum())
    
    # Removing duplicate rows which contain NaN fields.
    log('Removing rows with invalid numbers.')
    previous_dataset_length = len(pandas_dataframe)
    pandas_dataframe = pandas_dataframe.dropna()
    num_removed = previous_dataset_length - len(pandas_dataframe)
    log('New length: ' + str(len(pandas_dataframe)) + '. Removed: ' + str(num_removed) + ' rows.')
    
    return pandas_dataframe
    
def perform_numerical_analysis(pandas_dataframe, columns):
    log('Performing numerical analysis on the following columns: ' + ', '.join(columns))
    summary = pd.DataFrame({
        column: [
            np.mean(pandas_dataframe[column]), # Mean
            np.median(pandas_dataframe[column]), # Median
            np.std(pandas_dataframe[column]) # Standard deviation
        ]
        for column in columns
    }, index=['Mean', 'Median', 'Std'])

    print(summary)

def plot_histogram(pandas_dataframe):
    log('Creating a histogram of the data.')
    quality_scores = [3, 5, 6, 8]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, quality in zip(axes.flat, quality_scores):
        # Filter to get the quality 
        wines = pandas_dataframe[pandas_dataframe["quality"] == quality]
        wines["alcohol"].hist(bins=5, edgecolor="black", ax=ax)

        ax.set_title(f"Wine Quality {quality}")
        ax.set_xlabel("Alcohol Content (%)")
        ax.set_ylabel("Frequency")

    fig.suptitle("Alcohol Content Distribution by Wine Quality")
    fig.tight_layout()
    
    if verbose:
        plt.show()
    
    return fig

def multi_variable_scatterplot(pandas_dataframe):
    log('Creating a scatterplot of the data.')

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter_plot = ax.scatter(
        pandas_dataframe["alcohol"],
        pandas_dataframe["volatile acidity"],
        c=pandas_dataframe["quality"],
        cmap="viridis",
        s=25,
        alpha=0.65,
        edgecolors="none"
    )

    # Title and axis labels
    ax.set_title("Alcohol Content vs Volatile Acidity by Wine Quality", fontsize=14, pad=12)
    ax.set_xlabel("Alcohol Content (% ABV)", fontsize=11)
    ax.set_ylabel("Volatile Acidity (g/dm³)", fontsize=11)

    # Colourbar explains what the colours represent
    colourbar = fig.colorbar(scatter_plot, ax=ax)
    colourbar.set_label("Wine Quality Score", fontsize=11)

    # Improve readability
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()

    if verbose:
        plt.show()
    
    return scatter_plot

def seaborn_pair_plot(pandas_dataframe):
    log("Creating a pairplot of the data")
    
    columns = ["alcohol", "volatile acidity", "sulphates", "density", "pH", "quality"]
    plot_data = pandas_dataframe[columns]
    
    pairplot = sns.pairplot(
        plot_data,
        hue="quality",
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.5, "s": 15},
        diag_kws={"bins": 15}
    )
    pairplot.figure.suptitle("Plots of the Intersection Between Wine Characteristics and Quality", y=1.02)

    if verbose:
        plt.show()
        
    return pairplot

def main():
    parser = argparse.ArgumentParser(
        description="A simple project using the Python Standard Library, NumPy, Matplotlib Pyplot, Pandas to analyze datasets quickly and easily.",
        usage="python3 lucas_hautrive.py -p <path> -c [columns] {-v}",
        epilog='Example: python3 lucas_hautrive.py -p winequality_data.csv -v -c "fixed acidity" "volatile acidity" "citric acid"'
    )

    parser.add_argument('-p', '--path', type=str, default=DEFAULT_DATASET, help="File path to the dataset.")
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_NAMESPACE, help="The namespace used to name the output files.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose outputs enabled. To visualise charts/diagrams/plots, this flag must be set.")
    parser.add_argument('-e', '--export', action='store_true', help="Auto export charts/diagrams/plots.")
    parser.add_argument('-c', '--columns', type=str, nargs='+', default=DEFAULT_NUMERICAL_ANALYSIS_COLUMNS, help="Columns in the dataset to perform numerical analysis on.") 
    
    args = parser.parse_args()
    path = args.path
    columns = args.columns
    global verbose
    verbose = args.verbose
    is_exporting = args.export
    output_namespace = args.output
    
    # 1. Initial loading
    pandas_dataframe = load_dataset(path)
    if pandas_dataframe is None:
        log('A pandas dataframe couldnt be initialised. The program is stopping.')
        return
    
    # 2. Data cleaning
    pandas_dataframe = clean_dataset(pandas_dataframe)
    if pandas_dataframe is None:
        log('Failed to clean the dataset. The program is stopping.')
        return
    
    # 3. Numerical analysis
    perform_numerical_analysis(pandas_dataframe, columns)
    
    # 4. Simple plot
    histogram = plot_histogram(pandas_dataframe)
    if is_exporting:
        log('Exporting the histogram to file...')
        histogram.savefig(output_namespace + "histogram.png", dpi=300, bbox_inches="tight")
    
    # 5. Multi-variable scatterplot
    mv_scatterplot = multi_variable_scatterplot(pandas_dataframe)
    if is_exporting:
        log('Exporting the scatterplot to file...')
        mv_scatterplot.figure.savefig(output_namespace + "scatterplot.png", dpi=300, bbox_inches="tight")
    
    # 6. Extension task
    pair_plot = seaborn_pair_plot(pandas_dataframe)
    if is_exporting:
        log('Exporting the seaborn pairplot to file...')
        pair_plot.figure.savefig(output_namespace + "pairplot.png", dpi=300, bbox_inches="tight")
        
if __name__ == "__main__":
    main()