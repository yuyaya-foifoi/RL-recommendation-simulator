import logging

import torch
from torchnet.meter import AUCMeter

from configs.config import CFG_DICT
from src.utils.loss import get_loss_function
from src.utils.optimizer import get_optimizer

logger = logging.getLogger("src")


class Embedding_Trainer:
    def __init__(self, model_type: str, model, loaders, device: str):
        self.model_type = model_type
        self.model = model

        self.train_loader, self.val_loader, self.test_loader = loaders
        self.optimizer = self._get_optimizer()
        self.loss_func = self._get_loss_func()

        self.device = device

    def _get_optimizer(self):
        optimizer_params = {
            k: v
            for k, v in CFG_DICT[self.model_type]["OPTIMIZER"].items()
            if k != "name"
        }
        optimizer_algo = get_optimizer(
            CFG_DICT[self.model_type]["OPTIMIZER"]["name"]
        )
        optimizer = optimizer_algo(self.model.parameters(), **optimizer_params)
        logger.info(
            "Optimizer : {}".format(
                CFG_DICT[self.model_type]["OPTIMIZER"]["name"]
            )
        )
        return optimizer

    def _get_loss_func(self):
        loss_func = get_loss_function(CFG_DICT[self.model_type]["LOSS"])
        logger.info("Loss : {}".format(CFG_DICT[self.model_type]["LOSS"]))
        return loss_func

    def train(self):

        for epoch in range(CFG_DICT[self.model_type]["EPOCH"]):
            logger.info("{} training, epoch {}".format(self.model_type, epoch))

            loss, auc = self._train()
            logger.info("train / loss : {}, auc : {}".format(loss, auc))

            loss, auc = self._validation("val")
            logger.info("val / loss : {}, auc : {}".format(loss, auc))

            loss, auc = self._validation("test")
            logger.info("test / loss : {}, auc : {}".format(loss, auc))

            if auc > CFG_DICT[self.model_type]["AUC_THRESH"]:
                logger.info(
                    "AUC is {} and thresh is {} so break".format(
                        auc, CFG_DICT[self.model_type]["AUC_THRESH"]
                    )
                )
                return self.model

        return self.model

    def _train(self):

        self.model.train()
        auc = AUCMeter()
        sample_size = len(self.train_loader) * self.train_loader.batch_size

        loss_all = 0.0
        for batch in self.train_loader:
            user_feat, item_feat, history_feat, score = batch
            predict = self.model(
                user_feat.to(self.device),
                item_feat.to(self.device),
                history_feat.to(self.device),
            )

            auc.add(predict.detach(), score.to(self.device))
            loss = self.loss_func(predict, score.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_all += loss.item()

        return loss_all / sample_size, auc.value()[0]

    def _validation(self, data_type: str):

        if data_type == "val":
            loader = self.val_loader

        if data_type == "test":
            loader = self.test_loader

        loss_all = 0.0
        auc = AUCMeter()
        sample_size = len(loader) * loader.batch_size

        with torch.no_grad():
            self.model.eval()

            for batch in loader:
                user_feat, item_feat, history_feat, score = batch
                predict = self.model(
                    user_feat.to(self.device),
                    item_feat.to(self.device),
                    history_feat.to(self.device),
                )

                auc.add(predict.detach(), score.to(self.device))
                loss = self.loss_func(predict, score.to(self.device))

                loss_all += loss.item()

        return loss_all / sample_size, auc.value()[0]
