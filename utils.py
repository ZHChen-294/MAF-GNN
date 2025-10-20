import torch
import numpy as np

def accuracy(pred, label):
    """Compute accuracy."""
    pred, label = np.asarray(pred), np.asarray(label)
    return (pred == label).sum() / len(label)

def sensitivity(pred, label):
    """Compute sensitivity (recall for positive class)."""
    pred, label = np.asarray(pred), np.asarray(label)
    mask = (label == 1)
    return np.sum(pred[mask] == 1) / np.sum(mask) if np.sum(mask) else np.nan

def specificity(pred, label):
    """Compute specificity (recall for negative class)."""
    pred, label = np.asarray(pred), np.asarray(label)
    mask = (label == 0)
    return np.sum(pred[mask] == 0) / np.sum(mask) if np.sum(mask) else np.nan

def selecting_optim(args, model, lr, state=None):
    """Select optimizer."""
    name = args.optim.lower()
    print(f"Using optimizer: {args.optim}")

    if name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=args.betas, weight_decay=args.weight_decay)
    elif name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=args.betas, weight_decay=args.weight_decay)
    elif name == "radam":
        opt = torch.optim.RAdam(model.parameters(), lr=lr, betas=args.betas, weight_decay=args.weight_decay)
    elif name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optim}")

    if state is not None:
        opt.load_state_dict(state)
    return opt
