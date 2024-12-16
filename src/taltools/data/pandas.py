import pandas as pd
import numpy as np

def divide_segments(_df, label_col):
    df = _df.copy().reset_index()
    df['segment'] = (df[label_col] != df[label_col].shift()).cumsum()

    segments = df.groupby(['segment']).agg(
        label=(label_col, 'first'),
        start_frame=('frame', 'first'),
        end_frame=('frame', 'last')).reset_index(drop=True)
    segments['length'] = segments['end_frame'] - segments['start_frame']
    return segments

    # df['segment_change'] = df[label_col].ne(df[label_col].shift()).cumsum()
    # positive_segments = df[df[label_col] != 0].reset_index()
    #
    # segments = (
    #     positive_segments.groupby('segment_change')
    #     .agg(start_frame=('index', 'first'), end_frame=('index', 'last'))
    #     .reset_index(drop=True)
    # )
    # segments['length'] = segments['end_frame'] - segments['start_frame']
    # segments['label'] = 1
    # return segments

def fill_index(df, _from, _to):
    _missing = pd.DataFrame(index=list(set(range(_from, _to + 1)) - set(df.index)))
    return pd.concat([df, _missing], axis=0).sort_index()

