"""
SEMA-CL Jittor version - Main entry point.
"""
import setup_jittor_env  # Windows Jittor 环境初始化（必须在 import jittor 之前）
import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)
    train(args)


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='SEMA Continual Learning - Jittor version')
    parser.add_argument('--config', type=str, default='./exps/sema_inr_10task.json',
                        help='Json file of settings.')
    return parser


if __name__ == '__main__':
    main()
