import re
import numpy as np
import pandas as pd


def get_train_val(train_df):
    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    def expand_bbox(x):
        r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
        if len(r) == 0:
            r = [-1, -1, -1, -1]
        return r

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_df.drop(columns=['bbox'], inplace=True)
    train_df['x'] = train_df['x'].astype(np.float)
    train_df['y'] = train_df['y'].astype(np.float)
    train_df['w'] = train_df['w'].astype(np.float)
    train_df['h'] = train_df['h'].astype(np.float)

    image_ids = train_df['image_id'].unique()
    valid_ids = image_ids[-665:]
    train_ids = image_ids[:-665]

    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    train_df = train_df[train_df['image_id'].isin(train_ids)]

    return train_df, valid_df
