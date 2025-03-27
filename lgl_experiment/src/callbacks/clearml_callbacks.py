import os
import time
from typing import Optional, Any, List
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, \
    MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score

from src.loggers.clearml_logger import ClearMLLogger
from src.utils.callback_utils import LoggerNotFoundException, TrainerFastDevRunException


def get_clearml_logger(trainer: "pl.Trainer") -> Optional[ClearMLLogger]:
    if trainer.fast_dev_run:
        raise TrainerFastDevRunException(
            "Cannot use ClearML callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, ClearMLLogger):
        return trainer.logger

    if isinstance(trainer.loggers, list):
        for logger in trainer.loggers:
            if isinstance(logger, ClearMLLogger):
                return logger

    raise LoggerNotFoundException(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    def __init__(
        self,
        save_dir: Optional[str] = None,
    ):
        self.save_dir = save_dir

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        pth_model_path = os.path.join(self.save_dir, "last.pth")
        torch.save(trainer.model.to('cpu'), os.path.join(self.save_dir, pth_model_path))

        logger = get_clearml_logger(trainer)

        output_model = logger.new_output_model(
            name=type(pl_module).__name__,
            config_text=str(pl_module),
            framework="PyTorch",
        )
        output_model.update_weights(pth_model_path)

        named_modules = {k: str(v) for k, v in pl_module.named_modules()}

        logger = logger.task.logger
        for k, v in pl_module.state_dict().items():
            if len(v.size()) == 2:
                logger.report_matrix(
                    title=f"{k}: {named_modules[k.rsplit('.', 1)[0]]}",
                    series="weight",
                    matrix=v.cpu().numpy(),
                    iteration=0,
                )
            if len(v.size()) == 1:
                logger.report_vector(
                    title=f"{k}: {named_modules[k.rsplit('.', 1)[0]]}",
                    series="weight",
                    values=v.cpu().numpy(),
                    iteration=0,
                )


class LogMetricsAndRunningTime(Callback):
    def __init__(self, multi_label: bool = False, num_labels: int = 1, ignore_index: Optional[int] = None, device="cuda"):
        super().__init__()

        self.multi_label = multi_label
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.device = device

        self.preds = []
        self.targets = []
        self.training_epoch_times = []
        self.test_epoch_times = []

        self.acc: Optional[Metric] = None
        self.precision: Optional[Metric] = None
        self.recall: Optional[Metric] = None
        self.f1: Optional[Metric] = None

        self.training_epoch_start_time = None
        self.test_epoch_start_time = None

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # if self.device == "cuda":
        #     self.acc = torchmetrics.classification.BinaryAccuracy(ignore_index=self.ignore_index).cuda()
        #     self.precision = torchmetrics.classification.BinaryPrecision(ignore_index=self.ignore_index).cuda()
        #     self.recall = torchmetrics.classification.BinaryRecall(ignore_index=self.ignore_index).cuda()
        # elif self.device == "cpu":
        #     self.acc = torchmetrics.classification.BinaryAccuracy(ignore_index=self.ignore_index)
        #     self.precision = torchmetrics.classification.BinaryPrecision(ignore_index=self.ignore_index)
        #     self.recall = torchmetrics.classification.BinaryRecall(ignore_index=self.ignore_index)
        # else:
        #     assert False, "Unknown device"
        if self.multi_label:
            self.acc = MultilabelAccuracy(num_labels=self.num_labels, ignore_index=self.ignore_index).to(self.device)
            self.precision = MultilabelPrecision(num_labels=self.num_labels, ignore_index=self.ignore_index).to(self.device)
            self.recall = MultilabelRecall(num_labels=self.num_labels, ignore_index=self.ignore_index).to(self.device)
            self.f1 = MultilabelF1Score(num_labels=self.num_labels, ignore_index=self.ignore_index).to(self.device)
        else:
            self.acc = BinaryAccuracy(ignore_index=self.ignore_index).to(self.device)
            self.precision = BinaryPrecision(ignore_index=self.ignore_index).to(self.device)
            self.recall = BinaryRecall(ignore_index=self.ignore_index).to(self.device)
            self.f1 = BinaryF1Score(ignore_index=self.ignore_index).to(self.device)
        # print('LogMetricsAndRunningTime')

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.training_epoch_start_time = time.time()

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        unused: Optional = None,
    ) -> None:
        self.training_epoch_times.append(time.time() - self.training_epoch_start_time)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.preds.append(outputs["preds"])
        self.targets.append(outputs["targets"])

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.test_epoch_start_time = time.time()

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.test_epoch_times.append(time.time() - self.test_epoch_start_time)

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        logger = get_clearml_logger(trainer)
        log_data = {}
        for p, t in zip(self.preds, self.targets):
            self.acc.update(p, t)
            self.precision.update(p, t)
            self.recall.update(p, t)
            self.f1.update(p, t)
        log_data["test/Accuracy"] = self.acc.compute()
        log_data["test/Precision"] = self.precision.compute()
        log_data["test/Recall"] = self.recall.compute()
        log_data["test/F1"] = self.f1.compute()

        log_data["train/epoch_time"] = np.mean(self.training_epoch_times)
        log_data["test/epoch_time"] = np.mean(self.test_epoch_times)

        logger.log_metrics(log_data)
