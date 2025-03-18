import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d


def divide_segments(_df, label_col):
    df = _df.copy().reset_index()
    df['segment'] = (df[label_col] != df[label_col].shift()).cumsum()

    segments = df.groupby(['segment']).agg(
        label=(label_col, 'first'),
        start_frame=('frame', 'first'),
        end_frame=('frame', 'last')).reset_index(drop=True)
    segments['length'] = segments['end_frame'] - segments['start_frame']
    return segments

def fill_index(df, _from, _to):
    _missing = pd.DataFrame(index=list(set(range(_from, _to + 1)) - set(df.index)))
    return pd.concat([df, _missing], axis=0).sort_index()

def pd2np(df, columns, F, M):
    frame_indices = df['frame'].values
    person_indices = df['frame_offset'].values
    arr = np.zeros((F, M, len(columns)))
    vals = df[columns].to_numpy()
    arr[frame_indices, person_indices, :] = vals
    return arr


def pd2latex(df, caption="My Table", label="tab:mytable"):
    """
    Converts a Pandas DataFrame to a LaTeX table string.

    Parameters:
    df (pd.DataFrame): The dataframe to convert.
    caption (str): The caption for the table.
    label (str): The label for referencing the table.

    Returns:
    str: A LaTeX table string.
    """
    latex_str = df.to_latex(index=False, column_format="|c" * (len(df.columns) + 1) + "|", escape=False)

    latex_table = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\begin{tabular}{|l|" + "c|" * (len(df.columns) - 1) + "}\n"
                                                                    "\\hline\n"
            + latex_str +
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}"
    )

    return latex_table

def gaussian_smoothing(df, columns, sigma=2):
    smoothed_df = df.copy()
    for column in columns:
        smoothed_df[column] = gaussian_filter1d(smoothed_df[column].values, sigma=sigma)
    return smoothed_df

def assign_color(df, label_col, cmap):
    df[['r', 'g', 'b']] = np.array(df[label_col].map(cmap).tolist())
    return df

