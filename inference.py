CKPT_DIRS = [
    '../solution/8878/lstm_dual_attn_2_512/fold_1',
    '../solution/8879/lstm_dual_attn_2_512/fold_1/',
    '../solution/8880//lstm_dual_attn_2_512/fold_1/',
    '../solution/8881/lstm_dual_attn_4_256/fold_2/',    
    '../solution/8882/lstm_dual_attn_2_512/fold_1/',
    '../solution/8883/lstm_dual_attn_2_1024/fold_1/', 
    '../solution/8882_prev/lightning_logs/lstm_dual_attn_3_196/fold_8882/', 
    '../solution/8882_prev/lightning_logs/lstm_dual_attn_3_196/fold_8882/',
    '../solution/8883_prev_v2/lightning_logs/lstm_dual_attn_4_256/fold_88802',
    '../solution/8883_prev_v2/lightning_logs/lstm_dual_attn_4_256/fold_88802',
    
]
SUBM_DIR = '../submissions/'
MERGED_SUBM_NAME = '../submissions/FINAL.csv.gz'

from glob import glob
import gc
import json
import os
import random
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from src.test_callback import TestInferenceCallback

if not os.path.exists(SUBM_DIR):
    os.mkdir(SUBM_DIR)

from src.dataset.utils import (
    read_testcase_ids,
    read_sample,
    create_meta_mapping,
    read_metadata,
)
from src.dataset.test_ds import TestDataset
from src.dataset.train_ds import TrajectoryDataset
from src.metric import calculate_metric_on_batch
from src.modeling.lstm_dual_attn import LstmEncoderDecoderWithDualAttention
from src.modeling.prev_lstm_dual_attn import NoDPLstmEncoderDecoderWithDualAttention
import yaml
from src.lightning import TrajectoryLightningModule

import pandas as pd 
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    
    # Нормализация весов так, чтобы их сумма равнялась единице
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

test_ids = None

if __name__ == '__main__':
    for ckpt in CKPT_DIRS:
        hparams_path = f'{ckpt}/hparams.yaml'
        with open(hparams_path, "r") as f:
            hparams = yaml.safe_load(f)
        FOLD = hparams["FOLD"]
        NUM_LAYERS = hparams["NUM_LAYERS"]
        HIDDEN_SIZE = hparams["HIDDEN_SIZE"]
        LSTM_CLS_NAME = hparams["LSTM_CLS"]
        ATTN = hparams["ATTN"]
        SEED = hparams["SEED"]
        VAL_RATIO = hparams["VAL_RATIO"]
        LR = hparams["LR"]
        WD = hparams["WD"]
        DROPOUT = hparams.get("DROPOUT", 0)
        EPOCHS = hparams["EPOCHS"]
        BATCH_SIZE = hparams["BATCH_SIZE"]

        t_max = hparams.get("t_max", 100)
        warmup_epochs = hparams.get("warmup_epochs", 50)
        t_mult = hparams.get("t_mult", 1)
        scheduler = hparams.get("scheduler", "cosawr")
        scheduler_patience = hparams.get("scheduler_patience", 20)

        embedding_dim = hparams["embedding_dim"]
        localization_input_size = hparams["localization_input_size"]
        control_input_size = hparams["control_input_size"]
        DEVICE = hparams["DEVICE"]
        ROOT_DATA_FOLDER = hparams.get("DS_ROOT", '../YandexCup2024v2/')

        EXP_NAME = f"lstm_{ATTN}_{NUM_LAYERS}_{HIDDEN_SIZE}"

        seed_everything(SEED)

        TRAIN_DATASET_PATH = os.path.join(ROOT_DATA_FOLDER, "YaCupTrain")
        TEST_DATASET_PATH = os.path.join(ROOT_DATA_FOLDER, "YaCupTest")
        META_PATH = os.path.join(ROOT_DATA_FOLDER, "mapping.json")


        
        if not os.path.exists(META_PATH):
            train_ids = read_testcase_ids(TRAIN_DATASET_PATH)
            metas = []
            for testcase_id in tqdm(train_ids, desc="create meta mapping"):
                meta = read_metadata(
                    os.path.join(TRAIN_DATASET_PATH, str(testcase_id), "metadata.json")
                )
                metas.append(meta)
            mapping = create_meta_mapping(metas)

            with open(META_PATH, "w") as f:
                json.dump(mapping, f)

        mapping = json.load(open(META_PATH, "r"))
        for k, v in mapping.items():
            mapping[k]["map"] = {int(k): int(v) for k, v in v["map"].items()}
            mapping[k]["unk"] = int(v["unk"])
            assert v["unk"] in mapping[k]["map"]
        vehicle_feature_sizes = {k: len(v["map"]) for k, v in mapping.items()}

        if LSTM_CLS_NAME == "LstmEncoderDecoderWithDualAttention":
            LSTM_CLS = LstmEncoderDecoderWithDualAttention
        else:
            raise ValueError(f"Unknown model class: {LSTM_CLS_NAME}")

        if 'prev' in ckpt:
            LSTM_CLS = NoDPLstmEncoderDecoderWithDualAttention
        model = LSTM_CLS(
            vehicle_feature_sizes=vehicle_feature_sizes,
            embedding_dim=embedding_dim,
            localization_input_size=localization_input_size,
            control_input_size=control_input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)
        
        if test_ids is None:
            test_ids = read_testcase_ids(TEST_DATASET_PATH)
            test_dataset = TestDataset(TEST_DATASET_PATH, mapping, test_ids)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=None)

        versions = os.listdir(f'{ckpt}/checkpoints')
        for version in versions:
            if '8883_prev_v2' in ckpt:
                version = 'best_model-v1.ckpt' # i dont remember is 8883 fold have one broken model in best-models
            lightning_module = TrajectoryLightningModule.load_from_checkpoint(
                checkpoint_path=f'{ckpt}/checkpoints/{version}',
                model=model,
            )
            
            lightning_module.eval()
            lightning_module.model.eval()
            lightning_module.model.to(DEVICE)
            predictions = []
        
            with torch.inference_mode():
                for sample in tqdm(test_loader):
                    for k, v in sample.items():
                        sample[k] = v.to(DEVICE)
        
                    start_position = sample["start_position"].detach().cpu().numpy()[0]
                    requested_stamps = sample["requested_stamps"].detach().cpu().numpy()[0]
                    testcase_id = sample["testcase_id"].detach().cpu().item()
        
                    predicted_output_localization = lightning_module(sample)
        
                    time_steps = np.arange(5 * 1e9, 20 * 1e9, 4e7)
        
                    predicted_output_localization = (
                        predicted_output_localization.detach().cpu().numpy()[0]
                    )
                    predicted_output_localization[:, :3] += start_position
        
                    yaw_pred = predicted_output_localization[:, -1]
                    x_pred = predicted_output_localization[:, 0]
                    y_pred = predicted_output_localization[:, 1]
        
                    x_interp = np.interp(
                        requested_stamps, time_steps, x_pred, left=x_pred[0], right=x_pred[-1]
                    )
                    y_interp = np.interp(
                        requested_stamps, time_steps, y_pred, left=y_pred[0], right=y_pred[-1]
                    )
        
                    yaw_interp = np.interp(
                        requested_stamps,
                        time_steps,
                        yaw_pred,
                        left=yaw_pred[0],
                        right=yaw_pred[-1],
                    )
        
                    assert len(requested_stamps) == len(x_interp)
                    assert len(requested_stamps) == len(y_interp)
                    assert len(requested_stamps) == len(yaw_interp)
        
                    for stamp_ns, x, y, yaw in zip(
                        requested_stamps, x_interp, y_interp, yaw_interp
                    ):
                        predictions.append(
                            {
                                "testcase_id": int(testcase_id),
                                "stamp_ns": int(stamp_ns),
                                "x": x,
                                "y": y,
                                "yaw": yaw,
                            }
                        )
            predictions = pd.DataFrame(predictions)
            predictions["testcase_id"] = predictions["testcase_id"].apply(int)
            predictions = predictions.sort_values(by=["testcase_id", "stamp_ns"])
            predictions.to_csv(
                f"{SUBM_DIR}/{EXP_NAME}_fold_{FOLD}_seed_{SEED}_{version.split('.')[0]}.csv.gz",
                index=False,
                compression="gzip",
            )
        
    paths = glob(SUBM_DIR + '/*.csv.gz')
    merged = merge_submissions(paths)
    merged.to_csv(MERGED_SUBM_NAME, compression='gzip', index=False)

