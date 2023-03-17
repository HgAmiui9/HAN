import argparse
import platform
from dataclasses import dataclass


@dataclass
class Args:
    seed: int = 1
    batch_size: int = 32
    num_neighbors: int = 20
    lr: float = 0.001
    num_heads: list = [8]
    hidden_units: int = 8
    dropout: float = 0.6
    weight_decay: float = 0.001
    num_epochs: int = 100
    patience: int = 10
    dataset: str = "ACMRaw"
    device: str = "cpu"  # 默认使用CPU


def main(args: Args):
    # load data
    
    
    # initialize HAN model
    
    # epoch traning
    
    # evaluation
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mini-batch HAN")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_neighbors", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_heads", type=list, default=[8])
    parser.add_argument("--hidden_units", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="ACMRaw")

    # 根据不同操作系统设置不同设备
    sys_str = platform.system()
    if sys_str == "Linux":
        parser.add_argument("--device", type=str, default="cuda:0")
    elif sys_str == "Darwin":  # macOS
        parser.add_argument("--device", type=str, default="cpu")
    else:
        print("Unsupported system.")
        exit()

    args = Args(**vars(parser.parse_args()))
    main(args)