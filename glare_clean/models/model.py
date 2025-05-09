import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import F1Score, Accuracy, MatthewsCorrCoef
import json

from models import densenets, resnets
from helper.arguments import get_parameter

MODELS = {
    #
    "dense121": densenets.dense121,
    "dense161": densenets.dense161,
    "dense169": densenets.dense169,
    "dense201": densenets.dense201,
    #
    "res18": resnets.resnet18,
    "res34": resnets.resnet34,
    "res50": resnets.resnet50,
    "res101": resnets.resnet101,
    "res152": resnets.resnet152
    }


def get_model(id: str, num_classes: int, pretrained: str | bool, drop_rate: float):
    """
    Gets the respective model to the id

    Parameters
        id: str
            id of model
        num_classes: int
            number of classes
        pretrained: str | bool
            whether using on ImageNet pretrained weights or not

    """

    assert id in MODELS, f"Could not find model with id-number {id}"

    network_func = MODELS[id]

    if isinstance(pretrained, bool):
        pretrained = "IMAGENET1K_V1" if pretrained else None

    return network_func(num_classes=num_classes, pretrained=pretrained, drop_rate=drop_rate)


class Model(pl.LightningModule):
    def __init__(self, num_classes: int, spec: dict, weights_dir: str):
        """
        Init model architecture
        
        Parameters
        ----------
        num_classes: int
            number of classes
        spec: dict
            Dict containing important training specs to be saved after training

        """
        super().__init__()
        
        self.num_classes        = num_classes
        self.spec               = spec
        self.id                 = get_parameter(spec, "model", "dense169", str)
        self.pretrained         = get_parameter(spec, "pretrained", True, bool)
        self.drop_rate          = get_parameter(spec, "drop_rate", 0.0, float)
        self.model              = get_model(id=self.id, num_classes= self.num_classes, pretrained=self.pretrained, drop_rate=self.drop_rate)

        self.softmax            = nn.Softmax(dim=1)
        self.CE                 = nn.CrossEntropyLoss(reduction="none")
        self.lr                 = get_parameter(spec, "lr", 1e-6, float)
        self.use_lr_scheduler   = get_parameter(spec, "lr_scheduler", True, bool)
        self.lr_end_factor      = get_parameter(spec, "lr_end_factor", 0.001, float)
        self.epochs             = get_parameter(spec, "epochs", 25, int)

        ####
        self.weight_path        = weights_dir
        self.lr_list            = []
        ####
        
        self.validation_losses = [] 
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro') 
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self.mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters()

    """ def on_fit_start(self):
        print(f"Model is on device: {next(self.parameters()).device}") """
        
        
    def forward(self, x):
        """
        Forward the netword with inputs
        """
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        X, labels, ids, targets = batch
        loss, logits, logits_softmax = self._shared_step(X, labels)
        preds = torch.argmax(logits, dim=1)

        f1 = self.f1_score(preds, labels)
        acc = self.accuracy(preds, labels)
        mcc = self.mcc(preds, labels)
        
        self.log("train_loss", loss.detach().cpu(), prog_bar=False, on_epoch=True)
        self.log("lr", self._get_current_lr(), prog_bar=True, on_epoch=True)
        self.log("train_f1", f1.detach().cpu(), prog_bar=False, on_epoch=True)
        self.log("train_accuracy", acc.detach().cpu(), prog_bar=False, on_epoch=True)
        self.log("train_mcc", mcc.detach().cpu(), prog_bar=False, on_epoch=True)
        return loss
    

    def on_train_epoch_end(self, *args, **kwargs):
        os.makedirs(self.weight_path, exist_ok=True)
        torch.save(self.state_dict(), f'{self.weight_path}/epoch_{self.current_epoch}')
        self.lr_list.append(self._get_current_lr())
    

    def on_train_end(self):
        with open(f'{self.weight_path}/lr.json', "w") as f:
            json.dump(self.lr_list, f)
    

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        X, labels, ids, targets = batch
        loss, logits, logits_softmax = self._shared_step(X, labels)
            
        preds = torch.argmax(logits, dim=1)

        f1 = self.f1_score(preds, labels)
        acc = self.accuracy(preds, labels)
        mcc = self.mcc(preds, labels)
        #misclassification_rate = 1 - acc

        self.log("val_loss", loss.detach().cpu(), prog_bar=True, on_epoch=True)
        self.log("val_f1", f1.detach().cpu(), prog_bar=True, on_epoch=True)
        self.log("val_accuracy", acc.detach().cpu(), prog_bar=True, on_epoch=True)
        self.log("val_mcc", mcc.detach().cpu(), prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        print(f"Using lr: {self.lr}")
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.use_lr_scheduler:
            print("Init lr scheduler")
            lr_scheduler = {
                #'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3),
                'scheduler': optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=self.lr_end_factor, total_iters=self.epochs),  
                'interval': 'epoch'
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}
    

    def _shared_step(self, X, labels, detach2cpu: bool = False):
        logits = self(X)
        logits_softmax = self.softmax(logits)
        loss = torch.mean(self.CE(logits, labels))

        return loss, logits, logits_softmax

    

    def _get_current_lr(self):
        optimizer = self.trainer.optimizers[0]
        return optimizer.param_groups[0]['lr']








