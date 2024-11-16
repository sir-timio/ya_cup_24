import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import read_metadata, resample_data


class TestDataset(Dataset):
    def __init__(self, dataset_path: str, mapping: dict, testcase_ids=None):
        self.data = []
        self.mapping = mapping

        sampling_interval_ns = 4e7  # 0.04 seconds in nanoseconds
        initial_state_length = int(5 / 0.04)  # Steps for initial 5 seconds (125 steps)
        time_steps_localization = np.arange(0, 5 * 1e9, sampling_interval_ns)
        time_steps_control = np.arange(0, 20 * 1e9, sampling_interval_ns)

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
            requested_stamps = pd.read_csv(
                os.path.join(testcase_path, "requested_stamps.csv")
            )["stamp_ns"].values

            metadata = read_metadata(os.path.join(testcase_path, "metadata.json"))
            vehicle_features = self.encode_vehicle_features(metadata)
            control = pd.read_csv(os.path.join(testcase_path, "control.csv"))
            localization = pd.read_csv(os.path.join(testcase_path, "localization.csv"))

            localization_resampled = resample_data(
                localization["stamp_ns"].values,
                localization.drop(columns=["stamp_ns"]).values,
                time_steps_localization,
            )

            control_resampled = resample_data(
                control["stamp_ns"].values,
                control.drop(columns=["stamp_ns"]).values,
                time_steps_control,
            )

            input_localization = localization_resampled.copy()

            start_position = input_localization[0, :3].copy()
            input_localization[:, :3] -= start_position

            input_control_sequence = control_resampled[:initial_state_length].copy()
            output_control_sequence = control_resampled[initial_state_length:].copy()

            self.data.append(
                {
                    "testcase_id": int(testcase_id),
                    "requested_stamps": requested_stamps,
                    "start_position": start_position,
                    "vehicle_features": vehicle_features,
                    "input_localization": input_localization,
                    "input_control": input_control_sequence,
                    "output_control": output_control_sequence,
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
        tensor_dict = {}
        for k, v in sample.items():
            if k.startswith("vehicle"):
                tensor_dict[k] = torch.tensor(v, dtype=torch.long)
            elif k in ["testcase_id", "requested_stamps", "start_position"]:
                tensor_dict[k] = v
            else:
                tensor_dict[k] = torch.tensor(v, dtype=torch.float32)

        return tensor_dict
