import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies.colossalai import ColossalAIStrategy
import wandb

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel

import torch
from torch import Tensor


class ColossalEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)

    def _improvement_message(self, current: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                 f"Metric {self.monitor} improved by {abs(self.best_score - current)} >="
                 f" min_delta = {abs(self.min_delta)}. New best score: {current}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current}"
        return msg

    def _evaluate_stopping_criteria(self, current: Tensor):
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                 f"Monitored metric {self.monitor} = {current} is not finite."
                 f" Previous best value was {self.best_score}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                 "Stopping threshold reached:"
                 f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                 " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                 "Divergence threshold reached:"
                 f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                 " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                 should_stop = True
                 reason = (
                      f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                      f" Best score: {self.best_score}. Signaling Trainer to stop."
                 )

        return should_stop, reason


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     parser = pl.Trainer.add_argparse_args(parser)
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule
     model = ScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )

     # Set up logger configuration
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir="logs", name="tensorboard")
     else:
          logger = WandbLogger(project="sgmse", log_model=True, save_dir="logs")
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"logs/{logger.version}", save_last=True, filename='{epoch}-last'),
                  ColossalEarlyStopping(patience=10, monitor="valid_loss", mode="min")]
     if args.num_eval_files:
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"logs/{logger.version}", 
               save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
          checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"logs/{logger.version}", 
               save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
          callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          # strategy=DDPPlugin(find_unused_parameters=False),
          strategy=ColossalAIStrategy(placement_policy="auto",
                                      accelerator="gpu",
                                      min_chunk_size=32 * 1024**2,
                                      initial_scale=32),
          precision=16,
          logger=logger,
          log_every_n_steps=10,
          num_sanity_val_steps=0,
          callbacks=callbacks
     )

     # Train model
     trainer.fit(model)
