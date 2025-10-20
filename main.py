import os
import copy
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from utils import selecting_optim, accuracy, sensitivity, specificity
from data_loading import set_seed, multi_atlas_DataLoader
from Config import Config
from Model.MAF_GNN import MAF_GNN


# =========================================================
# Utility functions
# =========================================================
def init_weights(m):
    """Initialize Linear layers with Kaiming uniform initialization."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =========================================================
# Training function
# =========================================================
def train(model, train_loader, optimizer, loss_ce, epoch, path_save_info):
    model.train()
    train_epoch_cost = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for _, loader in enumerate(train_loader):
        optimizer.zero_grad()
        node_list, labels = loader
        node_list = [x.to(device) for x in node_list]
        labels = labels.to(device)
        adj_list = node_list

        logits = model(node_list, adj_list)
        loss = loss_ce(logits, labels)
        loss.backward()
        optimizer.step()

        train_epoch_cost += loss.item()
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(prob, axis=1)

        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(prob.tolist())
        all_preds.extend(preds.tolist())

    avg_loss = train_epoch_cost / len(train_loader)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy(all_preds, all_labels)
    sen = sensitivity(all_preds, all_labels)
    spec = specificity(all_preds, all_labels)

    try:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1], multi_class="ovr")
    except ValueError:
        auc = 0.0

    if epoch % 10 == 0:
        print(
            f"[Train] [Epoch:{epoch}] [Time:{time.time() - start_time:.1f}] "
            f"[Loss:{avg_loss:.4f}] [ACC:{acc:.4f}] [SEN:{sen:.4f}] "
            f"[SPEC:{spec:.4f}] [F1:{f1:.4f}] [AUC:{auc:.4f}]"
        )

    # Logging
    file_path = path_save_info.replace(".csv", "_train.csv")
    write_header = not os.path.exists(file_path)
    with open(file_path, "a") as f:
        if write_header:
            f.write("Epoch,Loss,Accuracy,Sensitivity,Specificity,F1,AUC\n")
        f.write(f"{epoch},{avg_loss},{acc},{sen},{spec},{f1},{auc}\n")


# =========================================================
# Testing function
# =========================================================
def test(model, test_loader, loss_ce, epoch, path_save_info):
    model.eval()
    test_epoch_cost = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for _, loader in enumerate(test_loader):
            node_list, labels = loader
            node_list = [x.to(device) for x in node_list]
            labels = labels.to(device)
            adj_list = node_list

            logits = model(node_list, adj_list)
            loss = loss_ce(logits, labels)
            test_epoch_cost += loss.item()

            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(prob, axis=1)

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(prob.tolist())
            all_preds.extend(preds.tolist())

    avg_loss = test_epoch_cost / len(test_loader)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy(all_preds, all_labels)
    sen = sensitivity(all_preds, all_labels)
    spec = specificity(all_preds, all_labels)

    try:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1], multi_class="ovr")
    except ValueError:
        auc = 0.0

    if epoch % 10 == 0:
        print(
            f"[Test] [Epoch:{epoch}] [Time:{time.time() - start_time:.1f}] "
            f"[Loss:{avg_loss:.4f}] [ACC:{acc:.4f}] [SEN:{sen:.4f}] "
            f"[SPEC:{spec:.4f}] [F1:{f1:.4f}] [AUC:{auc:.4f}]"
        )

    # Logging
    file_path = path_save_info.replace(".csv", "_test.csv")
    write_header = not os.path.exists(file_path)
    with open(file_path, "a") as f:
        if write_header:
            f.write("Epoch,Loss,Accuracy,Sensitivity,Specificity,F1,AUC\n")
        f.write(f"{epoch},{avg_loss},{acc},{sen},{spec},{f1},{auc}\n")

    return avg_loss, [acc, sen, spec, f1, auc]


# =========================================================
# Main execution
# =========================================================
if __name__ == "__main__":
    start_time = time.time()
    args = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    args.timestamp = args.model_save_timestamp.strip("_")
    args.fold_num = 1
    print(f"Fold: {args.fold_num} | Device: {device}")

    # Data loader
    train_loader, test_loader, weight = multi_atlas_DataLoader(args=args, atlases=args.Multi_atlas)

    # Model definition
    model = MAF_GNN(
        input_dim=args.Multi_numROI,
        hidden_dim=64,
        output_dim=2,
        num_inputs=len(args.Multi_numROI),
    ).to(device)
    model.apply(init_weights)

    # Save directory setup
    args.model_name = "MAF_GNN"
    args.save_dir = os.path.join(
        "F:/Dataset/Depression/DIRECT/UESTC/Result",
        args.dataset,
        args.Holistic_atlas,
        f"{args.model_name}_{args.timestamp}",
    )
    os.makedirs(args.save_dir, exist_ok=True)
    path_save_info = os.path.join(args.save_dir, f"log_info_{args.timestamp}_{args.fold_num}.csv")

    # Optimizer, scheduler, loss
    optimizer = selecting_optim(args=args, model=model, lr=args.lr)
    scheduler_model = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    loss_ce = nn.CrossEntropyLoss(weight=weight)

    # Track best model
    best_loss = float("inf")
    best_epoch = 0
    best_result = []
    best_state = None

    for epoch in range(args.num_epoch):
        train(model, train_loader, optimizer, loss_ce, epoch, path_save_info)
        test_loss, result = test(model, test_loader, loss_ce, epoch, path_save_info)

        # Save best model based on test loss
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_result = result

    print(f"\n✅ Best model found at epoch {best_epoch} with loss {best_loss:.4f}")
    print(
        "Performance: "
        f"[ACC:{best_result[0]:.4f}] [SEN:{best_result[1]:.4f}] "
        f"[SPEC:{best_result[2]:.4f}] [F1:{best_result[3]:.4f}] [AUC:{best_result[4]:.4f}]"
    )

    # Save best model and summary
    with open(path_save_info.replace(".csv", "_test.csv"), "a") as f:
        f.write(
            f"{best_epoch},{best_loss},{best_result[0]},{best_result[1]},"
            f"{best_result[2]},{best_result[3]},{best_result[4]}\n"
        )
    torch.save(best_state, path_save_info.replace(".csv", "_best_model.pth"))
