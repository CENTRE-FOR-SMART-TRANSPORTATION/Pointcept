"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import Trainer
from pointcept.engines.launch import launch
import json

def main_worker(cfg):
    # print()
    # print("In main_func now...")
    cfg = default_setup(cfg) # I think it is adding batch size and stuff
    trainer = Trainer(cfg)
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    print(args)
    # print(f"The chosen config file that will be passed to config parser is: {args.config_file}")
    cfg = default_config_parser(args.config_file, args.options)
    # print(f"Config converted from file, the model backbone type is", cfg.model.backbone.type)
    # print("Launching worker...")
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
