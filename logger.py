import tensorflow as tf
from pathlib import Path
import time
from typing import Union, Optional


class Logger:
    def __init__(self, tag: Optional[str] = None):
        tag = tag if tag is not None else str(int(time.time()))
        run_dir = Path("runs") / tag
        (run_dir / "logs").mkdir(parents=True)
        self.writer = tf.summary.create_file_writer(str(run_dir / "logs"))

    def log(self, msg: str, value: Union[int, float], step: int) -> None:
        with self.writer.as_default():
            tf.summary.scalar(msg, value, step)
