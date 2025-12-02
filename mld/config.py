import os
import importlib
from typing import Type, TypeVar
from argparse import ArgumentParser

from omegaconf import OmegaConf, DictConfig


def get_module_config(cfg_model: DictConfig, paths: list[str], cfg_root: str) -> DictConfig:
    files = [os.path.join(cfg_root, 'modules', p+'.yaml') for p in paths]
    for file in files:
        assert os.path.exists(file), f'{file} is not exists.'
        with open(file, 'r') as f:
            cfg_model.merge_with(OmegaConf.load(f))
    return cfg_model


def get_obj_from_str(string: str, reload: bool = False) -> Type:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: DictConfig) -> TypeVar:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default='configs/mld_t2m.yaml', help="The main config file")
    parser.add_argument('--example', type=str, required=False, help="The input texts and lengths with txt format")
    parser.add_argument('--example_hint', type=str, required=False, help="The input hint ids and lengths with txt format")
    parser.add_argument('--no-plot', action="store_true", required=False, help="Whether to plot the skeleton-based motion")
    parser.add_argument('--replication', type=int, default=1, help="The number of replications of sampling")
    parser.add_argument('--vis', type=str, default="tb", choices=['tb', 'swanlab'], help="The visualization backends: tensorboard or swanlab")
    parser.add_argument('--optimize', action='store_true', help="Enable optimization for motion control")


    parser.add_argument('--lambda_t2m', type=float, default=0)
    parser.add_argument('--lambda_m2m', type=float, default=0)
    parser.add_argument('--spm_path', type=str, default=None)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    cfg_root = os.path.dirname(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
    cfg = OmegaConf.merge(cfg, cfg_model)

    cfg.example = args.example
    cfg.example_hint = args.example_hint
    cfg.no_plot = args.no_plot
    cfg.replication = args.replication
    cfg.vis = args.vis
    cfg.optimize = args.optimize
    
    cfg.lambda_t2m = args.lambda_t2m
    cfg.lambda_m2m = args.lambda_m2m
    cfg.spm_path = args.spm_path

    return cfg

def parse_args_SPM() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="The main config file")

    parser.add_argument('--lambda_t2m', type=float, default=0, help='')
    parser.add_argument('--lambda_m2m', type=float, default=0, help='')   
    
    parser.add_argument('--NoiseThr', type=float, default=None, help='threshold')
    parser.add_argument('--step_aware', type=str, default=None, help='')
    parser.add_argument('--maxT', type=int, default=None, help='')
    parser.add_argument('--spm_path', type=str, default='None', help='')
    
    parser.add_argument('--CLThr', type=float, default=None, help='')
    parser.add_argument('--CLTemp', type=float, default=None, help='')
    
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    cfg_root = os.path.dirname(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
    cfg = OmegaConf.merge(cfg, cfg_model)
    
    if args.NoiseThr is not None:
        cfg.NoiseThr = args.NoiseThr
    if args.maxT is not None:
        cfg.maxT = args.maxT
    if args.step_aware is not None:
        cfg.step_aware = args.step_aware
    if args.CLThr is not None:
        cfg.CLThr = args.CLThr
    if args.CLTemp is not None:
        cfg.CLTemp = args.CLTemp
        
    cfg.lambda_t2m = args.lambda_t2m
    cfg.lambda_m2m = args.lambda_m2m
    
    
    
    return cfg

