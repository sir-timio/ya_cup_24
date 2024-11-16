import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np


class TestInferenceCallback(Callback):
    def __init__(
        self, test_loader, save_dir, interval=200, start_from=999, device="cuda"
    ):
        super().__init__()
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.interval = interval
        self.start_from = start_from
        self.device = device
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        if (current_epoch >= self.start_from) and (current_epoch % self.interval == 0):
            pl_module.model.eval()
            predictions = []
            with torch.inference_mode():
                for sample in tqdm(
                    self.test_loader, desc=f"Inference at epoch {current_epoch}"
                ):
                    sample = {k: v.to(self.device) for k, v in sample.items()}

                    start_position = sample["start_position"].detach().cpu().numpy()[0]
                    requested_stamps = (
                        sample["requested_stamps"].detach().cpu().numpy()[0]
                    )
                    testcase_id = sample["testcase_id"].detach().cpu().item()

                    predicted_output_localization = pl_module(sample)

                    time_steps = np.arange(5 * 1e9, 20 * 1e9, 4e7)

                    predicted_output_localization = (
                        predicted_output_localization.detach().cpu().numpy()[0]
                    )
                    predicted_output_localization[:, :3] += start_position

                    yaw_pred = predicted_output_localization[:, -1]
                    x_pred = predicted_output_localization[:, 0]
                    y_pred = predicted_output_localization[:, 1]

                    x_interp = np.interp(
                        requested_stamps,
                        time_steps,
                        x_pred,
                        left=x_pred[0],
                        right=x_pred[-1],
                    )
                    y_interp = np.interp(
                        requested_stamps,
                        time_steps,
                        y_pred,
                        left=y_pred[0],
                        right=y_pred[-1],
                    )

                    yaw_interp = np.interp(
                        requested_stamps,
                        time_steps,
                        yaw_pred,
                        left=yaw_pred[0],
                        right=yaw_pred[-1],
                    )

                    assert len(requested_stamps) == len(
                        x_interp
                    ), "Mismatch in x_interp length"
                    assert len(requested_stamps) == len(
                        y_interp
                    ), "Mismatch in y_interp length"
                    assert len(requested_stamps) == len(
                        yaw_interp
                    ), "Mismatch in yaw_interp length"

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

            predictions_df = pd.DataFrame(predictions)
            predictions_df["testcase_id"] = predictions_df["testcase_id"].astype(int)
            predictions_df = predictions_df.sort_values(by=["testcase_id", "stamp_ns"])

            submission_path = os.path.join(
                self.save_dir,
                f"submissions_epoch_{current_epoch}_fold_{trainer.logger.version}_seed_{trainer.logger.name}.csv.gz",
            )
            predictions_df.to_csv(
                submission_path,
                index=False,
                compression="gzip",
            )
            pl_module.model.train()
