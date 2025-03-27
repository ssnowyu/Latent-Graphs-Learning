from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union, Sequence, List
from clearml import Task, OutputModel
from clearml.utilities.proxy_object import verify_basic_type
from lightning_fabric.utilities.logger import (
    _convert_params,
    _flatten_dict,
    _sanitize_callable_params,
    _add_prefix,
)
from lightning_fabric.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.logger import _scan_checkpoints
from torch import Tensor


def _convert_params_to_basic_type(params: Dict[str, Any]) -> Dict[str, str]:
    return {k: str(v) if not verify_basic_type(v) else v for k, v in params.items()}


class ClearMLLogger(Logger):
    LOGGER_JOIN_CHAR = "-"
    METRIC_SPLIT_CHAR = "/"
    CHECKPOINTS_DIR = "checkpoints"

    def __init__(
        self,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        save_dir: str = ".",
        log_ckpt: bool = False,
        checkpoint_name: Optional[str] = None,
    ):
        Logger.__init__(self)
        self.task = Task.init(
            project_name=project_name,
            task_name=task_name,
            # auto_connect_frameworks={'pytorch': True},
            tags=tags,
            output_uri=save_dir,
        )
        self._logger = self.task.get_logger()
        self._log_ckpt = log_ckpt
        self._checkpoint_name = checkpoint_name
        self._save_dir = save_dir

        self._checkpoint_callback: Optional[ModelCheckpoint] = None

    def parse_raw_metrics(self, metrics: Dict[str, float]) -> List[Dict[str, float]]:
        # print(f"metrics is {metrics}")
        res = []
        for key, value in metrics.items():
            # step of batch or epoch
            if self.METRIC_SPLIT_CHAR not in key:
                continue
            title, series = key.split(self.METRIC_SPLIT_CHAR)
            res.append(
                {
                    "title": title,
                    "series": series,
                    "value": value,
                }
            )
        return res

    @property
    def name(self) -> Optional[str]:
        return self.task.name

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self.task.id

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any
    ) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        params = _convert_params_to_basic_type(params)
        self.task.connect(params, name="CL-hparams")

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = _add_prefix(metrics, "", self.LOGGER_JOIN_CHAR)
        if step is None:
            step = 0
        for params in self.parse_raw_metrics(metrics):
            self._logger.report_scalar(**dict(params, **{"iteration": step}))

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # log checkpoints as artifacts
        if self._log_ckpt:
            self._checkpoint_callback = checkpoint_callback

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if status != "success":
            # Currently, checkpoints only get logged on success
            return
        # log checkpoints as artifacts
        # checkpoints_dir = os.path.join(self._save_dir, "checkpoints")
        if self._checkpoint_callback is not None:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        checkpoints = _scan_checkpoints(checkpoint_callback, {})

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = "checkpoint"
            alias = "best" if p == checkpoint_callback.best_model_path else "latest"
            self.task.upload_artifact(
                f"{self._checkpoint_name}-{alias}",
                p,
                metadata=metadata,
                delete_after_upload=True,
            )

    # def watch_model(self, model_path: str):
    #     output_model = OutputModel(self.task, framework="PyTorch")
    #     output_model.update_weights(model_path)

    def new_output_model(self, **kwargs):
        return OutputModel(self.task, **kwargs)
