"""Train GeoTransformer or RTOR on P2I-LReg synthetic rigid pairs."""

import argparse
import json
import os
import os.path as osp
import sys


ROOT = osp.realpath(osp.join(osp.dirname(__file__), "..", ".."))
if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

import torch
import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer

from config import make_cfg
from dataset import train_valid_data_loader
from loss import OverallLoss
from model import create_model


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, parser, train_limit=None, validation_limit=None):
        super().__init__(
            cfg,
            max_epoch=cfg.optim.max_epoch,
            parser=parser,
            grad_acc_steps=cfg.optim.grad_acc_steps,
        )
        self.cfg = cfg
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(
            cfg,
            self.distributed,
            train_limit=train_limit,
            validation_limit=validation_limit,
        )
        self.register_loader(train_loader, val_loader)
        model = self.register_model(create_model(cfg).cuda())
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum,
            weight_decay=cfg.optim.weight_decay,
        )
        self.register_optimizer(optimizer)
        self.register_scheduler(
            optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        )
        self.loss_func = OverallLoss(cfg).cuda()
        protocol = {
            "name": cfg.protocol.name,
            "architecture": cfg.ablation.architecture,
            "rtor_enabled": bool(cfg.ablation.rtor_enabled),
            "a3_enabled": bool(cfg.ablation.a3_enabled),
            "neighbor_limits": [int(value) for value in neighbor_limits],
            "micro_batch_size": int(cfg.train.batch_size),
            "gradient_accumulation_steps": int(cfg.optim.grad_acc_steps),
            "effective_batch_size": int(cfg.protocol.effective_batch_size),
            "geotransformer_angle_k": int(cfg.geotransformer.angle_k),
            "geotransformer_hidden_dim": int(cfg.geotransformer.hidden_dim),
            "unit": cfg.protocol.unit,
            "transform": cfg.protocol.transform,
        }
        self.save_state("p2i_lreg_protocol", protocol)
        with open(osp.join(cfg.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"protocol": protocol, "command": sys.argv}, handle, indent=2)

    def _step(self, data):
        output = self.model(data)
        result = self.loss_func(output, data)
        if self.model.training and self.grad_acc_steps > 1:
            result["loss"] = result["loss"] / self.grad_acc_steps
        return output, result

    def train_step(self, epoch, iteration, data):
        return self._step(data)

    def val_step(self, epoch, iteration, data):
        return self._step(data)


def make_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--architecture", choices=["geotransformer", "rtor"], required=True)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--validation_limit", type=int, default=None)
    parser.add_argument("--neighbor_limits", type=int, nargs=4, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main():
    parser = make_parser()
    known, _ = parser.parse_known_args()
    cfg = make_cfg(known.architecture)
    if known.smoke:
        cfg.optim.max_epoch = 1
        known.train_limit = known.train_limit or 2
        known.validation_limit = known.validation_limit or 1
        cfg.data.neighbor_calibration_samples = 1
    if known.max_epoch is not None:
        if known.max_epoch <= 0:
            parser.error("--max_epoch must be positive")
        cfg.optim.max_epoch = known.max_epoch
    if known.train_limit is not None and known.train_limit < cfg.optim.grad_acc_steps:
        parser.error("--train_limit must cover at least one effective batch")
    if known.neighbor_limits is not None:
        cfg.data.neighbor_limits = known.neighbor_limits
    Trainer(cfg, parser, known.train_limit, known.validation_limit).run()


if __name__ == "__main__":
    main()
