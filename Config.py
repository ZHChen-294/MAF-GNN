import argparse
import torch
from datetime import datetime

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Configuration for MAF-GNN training')
        timestamp = datetime.today().strftime("%Y%m%d%H%M%S")

        # Device
        parser.add_argument("--cuda_num", type=str, default="0", help="GPU ID (0~5)")
        parser.add_argument("--device", help="torch device (auto-detected if not set)")

        # Training hyperparameters
        parser.add_argument("--seed", type=int, default=100, help="Random seed")
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs")
        parser.add_argument("--test_epoch_checkpoint", type=int, default=10, help="Test interval (epochs)")
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
        parser.add_argument("--optim", type=str, default="Adam", help="Optimizer type: Adam / AdamW / SGD / RAdam")
        parser.add_argument("--gamma", type=float, default=0.995, help="Learning rate decay factor")

        # Dataset and atlas settings
        parser.add_argument("--dataset", type=str, default="Data_615", help="Dataset name (Data_615 / Data_1570)")
        parser.add_argument("--Holistic_atlas", type=str, default="AHC",
                            help="Single atlas for holistic modeling (e.g., AH, AC, HC, AHC)")
        parser.add_argument("--Multi_atlas", nargs="+", default=["AAL", "Harvard", "Craddock"],
                            help="List of multiple atlases for multimodal fusion")
        parser.add_argument("--Multi_numROI", nargs="+", type=int, default=[116, 112, 200],
                            help="Number of ROIs for each atlas")

        self.args = parser.parse_args()

        # Device setup
        self.cuda_num = self.args.cuda_num
        self.device = torch.device(f"cuda:{self.cuda_num}" if torch.cuda.is_available() else "cpu")

        # Training
        self.seed = self.args.seed
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size
        self.num_epoch = self.args.num_epoch
        self.test_epoch_checkpoint = self.args.test_epoch_checkpoint
        self.weight_decay = self.args.weight_decay
        self.optim = self.args.optim
        self.gamma = self.args.gamma

        # Dataset
        self.dataset = self.args.dataset
        self.Holistic_atlas = self.args.Holistic_atlas
        self.Multi_atlas = self.args.Multi_atlas
        self.Multi_numROI = self.args.Multi_numROI

        # Timestamp (optional for saving)
        self.model_save_timestamp = f"_{timestamp}"
