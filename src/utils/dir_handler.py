import datetime
import logging
import os

import torch


class File_Handler:
    def __init__(self):
        self.date = self._get_date()
        self.log_dir = os.path.join("./log", self.date)
        self._mkdirs()

    def _get_date(self):
        str_date = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
        str_date = str_date.replace(":", "_").replace("-", "_")

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

    def save(self, file_object, save_name):
        path = os.path.join(self.log_dir, save_name)
        torch.save(file_object, path)
