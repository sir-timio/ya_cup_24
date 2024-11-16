import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import read_metadata, resample_data


class TrajectoryDataset(Dataset):
    def __init__(
        self, dataset_path: str, mapping: dict, testcase_ids=None, training=True
    ):
        self.data = []
        self.mapping = mapping
        self.training = training

        self.sampling_interval_ns = 4e7  # 0.04 seconds in nanoseconds
        self.initial_state_length = int(
            5 / 0.04
        )  # Steps for initial 5 seconds (125 steps)
        self.target_length = int(15 / 0.04)  # Steps from 5s to 20s (375 steps)
        self.sequence_length = (
            self.initial_state_length + self.target_length
        )  # Total steps (500 steps)
        self.time_steps = np.arange(
            0, 60 * 1e9, self.sampling_interval_ns
        )  # Target time steps

        self.control_lag_ns = 1e8  # Control lag of 100 ms in nanoseconds (adjustable)
        self.lag_steps = int(
            self.control_lag_ns / self.sampling_interval_ns
        )  # Number of steps corresponding to the lag
        # actually just 2.

        if testcase_ids is None:
            testcase_ids = sorted(
                [
                    name
                    for name in os.listdir(dataset_path)
                    if os.path.isdir(os.path.join(dataset_path, name))
                ]
            )

        for testcase_id in tqdm(testcase_ids):
            testcase_id = str(testcase_id)
            testcase_path = os.path.join(dataset_path, testcase_id)
            metadata = read_metadata(os.path.join(testcase_path, "metadata.json"))
            vehicle_features = self.encode_vehicle_features(metadata)

            # Load control and localization data
            control = pd.read_csv(os.path.join(testcase_path, "control.csv"))
            localization = pd.read_csv(os.path.join(testcase_path, "localization.csv"))

            # Resample with interpolation and fallback to nearest
            localization_resampled = resample_data(
                localization["stamp_ns"].values,
                localization.drop(columns=["stamp_ns"]).values,
                self.time_steps,
            )

            control_resampled = resample_data(
                control["stamp_ns"].values,
                control.drop(columns=["stamp_ns"]).values,
                self.time_steps,
            )
            max_start_idx = len(self.time_steps) - self.sequence_length

            self.data.append(
                {
                    "vehicle_features": vehicle_features,
                    "localization_resampled": localization_resampled,
                    "control_resampled": control_resampled,
                    "max_start_idx": max_start_idx,
                }
            )

    def encode_vehicle_features(self, metadata):
        feats = []
        for k, map_n_unk in self.mapping.items():
            feats.append(int(map_n_unk["map"].get(metadata[k], map_n_unk["unk"])))
        return feats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        vehicle_features = sample["vehicle_features"]

        if self.training:
            i = np.random.randint(0, sample["max_start_idx"])
        else:
            i = 0

        input_localization = deepcopy(
            sample["localization_resampled"][i : i + self.initial_state_length]
        )
        start_position = deepcopy(input_localization[0][:3])
        input_localization[:, :3] -= start_position  # Shift to zero

        # Target trajectory from t+5s to t+20s
        output_localization = deepcopy(
            sample["localization_resampled"][
                i + self.initial_state_length : i + self.sequence_length
            ]
        )
        output_localization[:, :3] -= start_position  # Shift to zero

        # Input and output control sequences
        input_control = deepcopy(
            sample["control_resampled"][i : i + self.initial_state_length]
        )
        output_control = deepcopy(
            sample["control_resampled"][
                i + self.initial_state_length : i + self.sequence_length
            ]
        )

        tensor_dict = {
            "vehicle_features": torch.tensor(vehicle_features, dtype=torch.long),
            "input_localization": torch.tensor(input_localization, dtype=torch.float32),
            "output_localization": torch.tensor(
                output_localization, dtype=torch.float32
            ),
            "input_control": torch.tensor(input_control, dtype=torch.float32),
            "output_control": torch.tensor(output_control, dtype=torch.float32),
        }
        return tensor_dict
