import datetime
import logging
import os
import shutil

import torch


class File_Handler:
    def __init__(self):
        self.date = self._get_date()
        self.log_dir = os.path.join("./log", self.date)
        self._mkdirs()
        self._save_config()

    def _get_date(self):
        now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
        str_date = str(now).split(".")[0].replace(" ", "_")
        str_date = str_date.replace(":", "_").replace("-", "_")
        return str_date

    def _mkdirs(self):
        return os.makedirs(self.log_dir, exist_ok=True)

    def make_logger_file(self):
        path = os.path.join(self.log_dir, "log_{}.txt".format(self.date))
        logger = logging.getLogger("src")
        hdlr = logging.FileHandler(path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
        return logger

    def _save_config(self):
        shutil.copy("./configs/config.py", self.log_dir)

    def save(self, file, name):
        path = os.path.join(self.log_dir, name)
        torch.save(file, path)

    def load(self, name):
        path = os.path.join(self.log_dir, name)
        return torch.load(path)
