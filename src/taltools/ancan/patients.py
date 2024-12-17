from datetime import datetime

import numpy as np
import pandas as pd


def str2date(date_str):
    if len(date_str) == 6:
        try:
            return datetime.strptime(date_str, '%d%m%y')
        except ValueError:
            pass
        return datetime.strptime(date_str, '%d%m%Y')
    elif len(date_str) == 8:
        return datetime.strptime(date_str, '%d%m%Y')
    return None


def add_patient_info(df, video_name_col, cinf=None):
    df['child_key'] = df[video_name_col].apply(lambda v: v.split('_')[0]).astype(int)
    df['assessment'] = df[video_name_col].apply(lambda v: '_'.join(v.split('_')[:-2]))
    if cinf is None:
        cinf = pd.read_csv(r'Z:\Users\TalBarami\241031_children_info.csv')
        cols = [0, 3, 4, 5]
        cinf = cinf.iloc[:, cols].dropna()
        cinf.columns = ['child_key', 'gender', 'date_of_birth', 'location']
        cinf['child_key'] = cinf['child_key'].astype(int)
        cinf['date_of_birth'] = pd.to_datetime(cinf['date_of_birth'], dayfirst=True)
        cinf.set_index('child_key', inplace=True)
    df['assessment_date'] = df['assessment'].apply(lambda s: str2date(s.split('_')[-1]))
    df['location'] = df['child_key'].apply(lambda x: cinf.loc[x, 'location'] if x in cinf.index else np.nan)
    df['gender'] = df['child_key'].apply(lambda x: cinf.loc[x, 'gender'] if x in cinf.index else np.nan)
    df['date_of_birth'] = df['child_key'].apply(lambda x: cinf.loc[x, 'date_of_birth'] if x in cinf.index else np.nan)
    df['age'] = (df['assessment_date'] - df['date_of_birth']).dt.days / 365.25
    bins = [0, 2, 4, 6, 8, 10, 100]
    labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10+']
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels)
    return df