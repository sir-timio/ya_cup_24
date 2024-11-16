SUBM_DIR = '../submissions/'
MERGED_SUBM_NAME = '../submissions/all_trained_merge.csv.gz'

from glob import glob
import json
import os

import numpy as np
import pandas as pd

from tqdm import tqdm

if not os.path.exists(SUBM_DIR):
    os.mkdir(SUBM_DIR)

import pandas as pd 
import numpy as np


def merge_submissions(paths: list, weights: list = None) -> pd.DataFrame:
    """
    Merge multiple submission files with given weights.

    Args:
        paths (list): List of paths to submission CSV files.
        weights (list): List of weights corresponding to each submission.

    Returns:
        pd.DataFrame: Merged predictions DataFrame.
    """
    if weights is None:
        weights = np.ones(len(paths))
        
    if len(paths) != len(weights):
        raise ValueError("Number of paths and weights must be the same.")
    
    weights = np.array(weights, dtype=np.float32)
    weight_sum = weights.sum()
    if weight_sum == 0:
        raise ValueError("Sum of weights must not be zero.")
    weights /= weight_sum
    print(weights)
    
    subms = []
    for path, weight in zip(paths, weights):
        df = pd.read_csv(path, compression='gzip')
        df[['x', 'y', 'yaw']] = df[['x', 'y', 'yaw']].astype(float)
        df[['x', 'y', 'yaw']] = df[['x', 'y', 'yaw']] * weight
        subms.append(df)
    
    concat_df = pd.concat(subms, ignore_index=True)
    
    merged_predictions = concat_df.groupby(['testcase_id', 'stamp_ns'], as_index=False).sum()
    
    return merged_predictions



if __name__ == '__main__':
    subm_paths = glob(SUBM_DIR + '/*.csv.gz')
    subm = merge_submissions(subm_paths)
    subm.to_csv(MERGED_SUBM_NAME, compression='gzip', index=False)