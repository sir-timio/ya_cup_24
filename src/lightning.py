import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from src.metric import calculate_metric_on_batch
from src.modeling.losses import WeightedLoss


class TrajectoryLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=1e-3,
        sceduler_type="cosawr",
        t_max=50,
        eta_min=1e-6,
        warmup_epochs=5,
        t_mult=1,
        scheduler_patience=25,
    ):
        super(TrajectoryLightningModule, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sceduler_type = sceduler_type

        self.scheduler_patience = scheduler_patience

        self.t_max = t_max
        self.t_mult = t_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        base_criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)
        weights = np.array([3, 3, 1, 1, 1, 3], dtype="float")
        weights /= np.linalg.norm(weights)
        self.criterion = WeightedLoss(base_criterion, weights)

    def forward(self, batch):
        vehicle_features = batch["vehicle_features"]
        input_localization = batch["input_localization"]
        input_control = batch["input_control"]
        output_control = batch["output_control"]

        predicted_output_localization = self.model(
            vehicle_features,
            input_localization,
            input_control,
            output_control,
        )

        return predicted_output_localization

    def training_step(self, batch, batch_idx):
        output_localization = batch["output_localization"]
        predicted_output_localization = self.forward(batch)

        loss = self.criterion(predicted_output_localization, output_localization)

        predicted_x_y_yaw = (
            predicted_output_localization[..., [0, 1, -1]].detach().cpu().numpy()
        )
        gt_x_y_yaw = output_localization[..., [0, 1, -1]].detach().cpu().numpy()
        batch_metric = calculate_metric_on_batch(predicted_x_y_yaw, gt_x_y_yaw)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_metric", batch_metric, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        output_localization = batch["output_localization"]
        predicted_output_localization = self.forward(batch)

        loss = self.criterion(predicted_output_localization, output_localization)

        predicted_x_y_yaw = (
            predicted_output_localization[..., [0, 1, -1]].detach().cpu().numpy()
        )
        gt_x_y_yaw = output_localization[..., [0, 1, -1]].detach().cpu().numpy()
        batch_metric = calculate_metric_on_batch(predicted_x_y_yaw, gt_x_y_yaw)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_metric", batch_metric, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.sceduler_type == "cosawr":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.t_max, T_mult=self.t_mult, eta_min=self.eta_min
                ),
                "name": "cosine_annealing_warm_restarts",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.9, patience=self.scheduler_patience, verbose=True
            )

        return [optimizer], [scheduler]
