import datetime
import logging
import os
import shutil

import torch
from pytz import timezone


class File_Handler:
    def __init__(self, log_dir=None):
        self.date = self._get_date()

        if log_dir is None:
            self.log_dir = os.path.join("./log", self.date)
        else:
            self.log_dir = os.path.join("./log", log_dir)

        self._mkdirs()
        self._save_config()

    def _get_date(self):
        now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
        str_date = str(now).split(".")[0].replace(" ", "_")
        str_date = str_date.replace(":", "_").replace("-", "_")
        return str_date

    def _mkdirs(self):
        return os.makedirs(self.log_dir, exist_ok=True)

    def _customTime(self, *args):
        return datetime.datetime.now(timezone("Asia/Tokyo")).timetuple()

    def _get_logging_file_handler(self):
        path = os.path.join(self.log_dir, "log_{}.txt".format(self.date))
        file_handler = logging.FileHandler(path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        formatter.converter = self._customTime
        file_handler.setFormatter(formatter)
        return file_handler

    def make_logger_file(self, name):
        logging.basicConfig()
        logger = logging.getLogger(name)
        file_handler = self._get_logging_file_handler()
        logger.addHandler(file_handler)
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
