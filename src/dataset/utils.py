import os
from collections import Counter
import numpy as np
import json
import pandas as pd


def read_testcase_ids(dataset_path: str):
    ids = sorted(
        [
            int(case_id)
            for case_id in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, case_id))
        ]
    )
    return ids


def read_metadata(metadata_path: str):
    with open(metadata_path, "r") as f:
        data = json.load(f)
    data["tires_front"] = data["tires"]["front"]
    data["tires_rear"] = data["tires"]["rear"]
    data.pop("tires")
    return data


def create_meta_mapping(metas):
    mapping = {
        "vehicle_id": [],
        "vehicle_model": [],
        "vehicle_model_modification": [],
        "location_reference_point_id": [],
        "tires_front": [],
        "tires_rear": [],
    }

    for meta in metas:
        for k in mapping:
            mapping[k].append(meta[k])

    for k, lst in mapping.items():
        unk = Counter(lst).most_common()[0][0]

        v = sorted(set(lst))
        map = {str(label): idx for idx, label in enumerate(v)}
        mapping[k] = {"map": map, "unk": str(unk)}
    return mapping


def read_sample(dataset_path, sample_id, is_test=False):
    sample_id = str(sample_id)
    sample = {}
    sample["localization"] = pd.read_csv(
        os.path.join(dataset_path, sample_id, "localization.csv")
    )
    sample["control"] = pd.read_csv(
        os.path.join(dataset_path, sample_id, "control.csv")
    )
    sample["metadata"] = read_metadata(
        os.path.join(dataset_path, sample_id, "metadata.json")
    )
    if is_test:
        sample["requested_stamps"] = pd.read_csv(
            os.path.join(dataset_path, sample_id, "requested_stamps.csv")
        )
    return sample


def resample_data(original_timestamps, original_values, target_timestamps):
    interpolated = np.zeros((len(target_timestamps), original_values.shape[1]))
    for i in range(original_values.shape[1]):
        interpolated[:, i] = np.interp(
            target_timestamps, original_timestamps, original_values[:, i]
        )

        # Fill values outside the original timestamp range with nearest values
        interpolated[target_timestamps < original_timestamps[0], i] = original_values[
            :, i
        ][0]
        interpolated[target_timestamps > original_timestamps[-1], i] = original_values[
            :, i
        ][-1]

    return interpolated
