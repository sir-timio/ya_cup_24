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

from src.modeling import (
    LstmEncoderDecoder,
    LstmEncoderDecoderWithAttention,
    LstmEncoderDecoderWithLuongAttention,
    LstmEncoderDecoderWithDualAttention,
)

model_classes = {
    "LstmEncoderDecoder": LstmEncoderDecoder,
    "LstmEncoderDecoderWithAttention": LstmEncoderDecoderWithAttention,
    "LstmEncoderDecoderWithLuongAttention": LstmEncoderDecoderWithLuongAttention,
    "LstmEncoderDecoderWithDualAttention": LstmEncoderDecoderWithDualAttention,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Trajectory Prediction Model")
    parser.add_argument(
        "--hparams",
        type=str,
        required=True,
        help="Path to the hparams.yaml file containing hyperparameters.",
    )
    return parser.parse_args()


SUBM_DIR = "../submissions"



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
# from src.modeling.lstm_dual_attn import LstmEncoderDecoderWithDualAttention

# from src.modeling import (
#     LstmEncoderDecoder,
#     LstmEncoderDecoderWithAttention,
#     LstmEncoderDecoderWithLuongAttention,
#     LstmEncoderDecoderWithDualAttention,
# )
import yaml
from src.lightning import TrajectoryLightningModule


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    hparams_path = args.hparams

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
    DROPOUT = hparams["DROPOUT"]
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
    ROOT_DATA_FOLDER = hparams["DS_ROOT"]

    START_FROM = hparams.get("START_FROM", 2000)
    INTERVAL = hparams.get("INTERVAL", 250)

    EXP_NAME = f"lstm_{ATTN}_{NUM_LAYERS}_{HIDDEN_SIZE}"

    seed_everything(SEED)

    TRAIN_DATASET_PATH = os.path.join(ROOT_DATA_FOLDER, "YaCupTrain")
    TEST_DATASET_PATH = os.path.join(ROOT_DATA_FOLDER, "YaCupTest")
    META_PATH = os.path.join(ROOT_DATA_FOLDER, "mapping.json")

    train_ids = read_testcase_ids(TRAIN_DATASET_PATH)
    print(len(train_ids))
    test_ids = read_testcase_ids(TEST_DATASET_PATH)
    print(len(test_ids))

    random.shuffle(train_ids)
    val_size = int(len(train_ids) * VAL_RATIO)
    val_identifiers = set(train_ids[:val_size])
    train_identifiers = set(train_ids[val_size:])

    if not os.path.exists(META_PATH):
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

    LSTM_CLS = model_classes.get(LSTM_CLS_NAME)
    if LSTM_CLS is None:
        raise ValueError(f"Unknown model class: {LSTM_CLS_NAME}")

    model = LSTM_CLS(
        vehicle_feature_sizes=vehicle_feature_sizes,
        embedding_dim=embedding_dim,
        localization_input_size=localization_input_size,
        control_input_size=control_input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    train_dataset = TrajectoryDataset(
        TRAIN_DATASET_PATH,
        mapping,
        testcase_ids=list(train_identifiers)[:],
        training=True,
    )
    val_dataset = TrajectoryDataset(
        TRAIN_DATASET_PATH,
        mapping,
        testcase_ids=list(val_identifiers)[:],
        training=False,
    )
    test_dataset = TestDataset(TEST_DATASET_PATH, mapping, test_ids)
    num_workers = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=None)

    logger = CSVLogger("logs", name=EXP_NAME, version=f"fold_{FOLD}")
    logger.log_hyperparams(hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", filename="best_model", save_top_k=3, mode="min"
    )

    inference_callback = TestInferenceCallback(
        test_loader=test_loader,
        save_dir=SUBM_DIR,
        start_from=START_FROM,
        interval=INTERVAL,
        device=DEVICE,
    )

    callbacks = [
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        inference_callback,
    ]

    lightning_module = TrajectoryLightningModule(
        model=model,
        learning_rate=LR,
        weight_decay=WD,
        t_max=t_max,
        warmup_epochs=warmup_epochs,
        t_mult=t_mult,
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=EPOCHS,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )
    trainer.fit(lightning_module, train_loader, val_loader)

    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    lightning_module = TrajectoryLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path, model=model
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
        f"{SUBM_DIR}/{EXP_NAME}_fold_{FOLD}_seed_{SEED}.csv.gz",
        index=False,
        compression="gzip",
    )
