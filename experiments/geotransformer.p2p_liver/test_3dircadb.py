#!/usr/bin/env python3
"""Evaluate the pretrained RTOR model on canonical 3D-IRCADb pairs."""

from __future__ import annotations

import argparse
import os
import os.path as osp
from pathlib import Path
import sys
import time

ROOT = osp.realpath(osp.join(osp.dirname(__file__), "..", ".."))
if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)
METRICS_ROOT = os.environ.get("P2P_METRICS_ROOT", ROOT)
if METRICS_ROOT not in sys.path:
    sys.path.append(METRICS_ROOT)

import torch

from geotransformer.engine import SingleTester
from geotransformer.utils.torch import release_cuda

from config import make_cfg
from ircadb_dataset import build_ircadb_loader, estimate_to_mm
from loss import Evaluator
from model import create_model
from tools.ircadb_benchmark import PredictionWriter


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility", choices=("all", "0.20", "0.30"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--architecture", choices=("geotransformer", "rtor_only", "a3_only", "rtor_a3"), default="rtor_a3")
    parser.add_argument("--registration-profile", choices=("legacy", "tight", "robust"), default="legacy")
    parser.add_argument("--dual-encoder", action="store_true")
    parser.add_argument("--interaction-profile", choices=("legacy", "cooperative", "soft_overlap"), default="cooperative")
    return parser


class Tester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        self.checkpoint = torch.load(self.args.snapshot, map_location="cpu", weights_only=False)
        protocol = self.checkpoint.get("metadata", {}).get("p2p_protocol", {})
        neighbor_limits = protocol.get("neighbor_limits", cfg.data.neighbor_limits)
        self.dataset, loader, limits = build_ircadb_loader(
            cfg, self.args.data_root, self.args.visibility, self.args.limit,
            self.args.num_workers, neighbor_limits,
        )
        self.logger.info(f"3D-IRCADb neighbor limits: {limits}")
        self.register_loader(loader)
        self.register_model(create_model(cfg).cuda())
        self.evaluator = Evaluator(cfg).cuda()
        self.writer = PredictionWriter(
            "ours", self.args.output, [record.sample_id for record in self.dataset.records]
        )
        self.started = 0.0

    def load_snapshot(self, snapshot):
        self.model.load_state_dict(self.checkpoint["model"], strict=True)
        del self.checkpoint

    def before_test_step(self, iteration, data_dict):
        self.started = time.perf_counter()

    def test_step(self, iteration, data_dict):
        return self.model(data_dict)

    def eval_step(self, iteration, data_dict, output_dict):
        result = self.evaluator(output_dict, data_dict)
        return {"PIR": result["PIR"], "IR": result["IR"]}

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        context = self.dataset.load_context(iteration - 1)
        estimate = estimate_to_mm(release_cuda(output_dict["estimated_transform"]), context)
        self.writer.write_success(
            context.record, estimate, context.gt_transform_mm,
            time.perf_counter() - self.started, context.source_mm, context.target_mm,
            pir=float(result_dict["PIR"]), ir=float(result_dict["IR"]),
        )

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return f"{self.dataset.records[iteration - 1].sample_id}"

    def after_test_epoch(self):
        payload = self.writer.finalize()
        self.logger.critical(f"Wrote {payload['counts']['total']} 3D-IRCADb predictions")


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--architecture", choices=("geotransformer", "rtor_only", "a3_only", "rtor_a3"), default="rtor_a3")
    pre_parser.add_argument("--registration-profile", choices=("legacy", "tight", "robust"), default="legacy")
    pre_parser.add_argument("--dual-encoder", action="store_true")
    pre_parser.add_argument("--interaction-profile", choices=("legacy", "cooperative", "soft_overlap"), default="cooperative")
    known, _ = pre_parser.parse_known_args()
    parser = make_parser()
    cfg = make_cfg(
        architecture=known.architecture,
        registration_profile=known.registration_profile,
        dual_encoder=known.dual_encoder,
        interaction_profile=known.interaction_profile,
    )
    Tester(cfg, parser).run()


if __name__ == "__main__":
    main()
