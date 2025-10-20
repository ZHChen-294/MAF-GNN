import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from scipy import io
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utility functions
# =========================================================
def set_seed(seed: int):
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and Python.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_subject_id(file_name: str):
    """
    Parse subject ID and diagnostic label from the file name.

    Parameters
    ----------
    file_name : str
        Example: 'ROISignals_S1-1-0001.mat'

    Returns
    -------
    symptom : str
        Diagnostic category ('MDD_Data' or 'NC')
    subject_ID : str
        Extracted subject ID (e.g., 'S1-1-0001')
    mat_full_name : str
        File name with .mat extension
    """
    mat_full_name = str(file_name)
    file_name_label = file_name.split("-")[1]

    if os.path.splitext(mat_full_name)[1] == ".npy":
        mat_full_name = mat_full_name.replace(".npy", ".mat")

    if file_name_label == "1":
        symptom = "MDD_Data"
        subject_ID = file_name[11:-4]
    elif file_name_label == "2":
        symptom = "NC"
        subject_ID = file_name[11:-4]
    else:
        print("Error: Unrecognized label in filename.")
        sys.exit()

    return symptom, subject_ID, mat_full_name


def get_FC_map(sub_list, dataset, atlas="AAL", fold_num=1):
    """
    Load functional connectivity (FC) matrices and labels for a given dataset and atlas.

    Parameters
    ----------
    sub_list : list of str
        List of subject filenames.
    dataset : str
        Dataset name (e.g., 'Data_615').
    atlas : str, optional
        Brain atlas name. Default: 'AAL'.
    fold_num : int, optional
        Fold number for cross-validation tracking.

    Returns
    -------
    data : np.ndarray
        Array of FC matrices (subjects × ROIs × ROIs).
    label : np.ndarray
        Binary labels (0=HC, 1=MDD).
    train_weight_index : list
        Counts of [HC, MDD] for class balancing.
    """
    file_dir = "F:/Dataset/Depression/DIRECT/UESTC"
    data_load_path = os.path.join(file_dir, dataset, atlas, f"MDD_{atlas}_FC")

    data, label = [], []
    train_weight_index = [0, 0]

    for file_name in sub_list:
        symptom, subject_ID, mat_full_name = get_subject_id(file_name)
        mat_path = os.path.join(data_load_path, mat_full_name)

        if not os.path.exists(mat_path):
            print(f"⚠️ Missing file: {mat_path}")
            continue

        mat = io.loadmat(mat_path)["ROI_Functional_connectivity"]
        mat[np.isinf(mat)] = 1.0  # Replace Inf with 1.0 for stability

        if symptom == "MDD_Data":
            label.append(1)
            train_weight_index[1] += 1
        else:
            label.append(0)
            train_weight_index[0] += 1

        data.append(mat)

    return np.array(data), np.array(label), train_weight_index


# =========================================================
# Dataset definition
# =========================================================
class MultiFunctionalConnectivityDataset(Dataset):
    """
    Custom Dataset for multi-atlas functional connectivity data.
    Each sample contains three FC matrices and a corresponding label.
    """

    def __init__(self, T1_tensor, T2_tensor, T3_tensor, label_tensor):
        self.T1_x = T1_tensor
        self.T2_x = T2_tensor
        self.T3_x = T3_tensor
        self.label = label_tensor

    def __len__(self):
        return len(self.T1_x)

    def __getitem__(self, idx):
        return [self.T1_x[idx], self.T2_x[idx], self.T3_x[idx]], self.label[idx]


# =========================================================
# Multi-atlas DataLoader (3 modalities)
# =========================================================
def multi_atlas_DataLoader(args, atlases):
    """
    Construct training and testing DataLoaders for 3-atlas (multi-view) FC data.

    Parameters
    ----------
    args : argparse.Namespace or Config
        Experimental configuration object containing dataset paths and parameters.
    atlases : list of str
        List of atlas names (length must be 3).

    Returns
    -------
    train_loader, test_loader : torch.utils.data.DataLoader
        PyTorch DataLoaders for training and testing.
    weight : torch.Tensor
        Class-balancing weight tensor for CrossEntropyLoss.
    """
    train_csv = f"F:/Dataset/Depression/DIRECT/UESTC/Data_csv_list/{args.dataset}/MDD_train_data_list_{args.fold_num}.csv"
    test_csv = f"F:/Dataset/Depression/DIRECT/UESTC/Data_csv_list/{args.dataset}/MDD_test_data_list_{args.fold_num}.csv"

    case_train_list = pd.read_csv(train_csv)["Subject ID"].tolist()
    case_test_list = pd.read_csv(test_csv)["Subject ID"].tolist()

    # Load FC data for each atlas
    train_data_1, train_label, train_weight_index = get_FC_map(case_train_list, args.dataset, atlas=atlases[0])
    test_data_1, test_label, _ = get_FC_map(case_test_list, args.dataset, atlas=atlases[0])

    train_data_2, _, _ = get_FC_map(case_train_list, args.dataset, atlas=atlases[1])
    test_data_2, _, _ = get_FC_map(case_test_list, args.dataset, atlas=atlases[1])

    train_data_3, _, _ = get_FC_map(case_train_list, args.dataset, atlas=atlases[2])
    test_data_3, _, _ = get_FC_map(case_test_list, args.dataset, atlas=atlases[2])

    # Convert to tensors
    train_data_1 = torch.FloatTensor(train_data_1).to(args.device)
    test_data_1 = torch.FloatTensor(test_data_1).to(args.device)
    train_data_2 = torch.FloatTensor(train_data_2).to(args.device)
    test_data_2 = torch.FloatTensor(test_data_2).to(args.device)
    train_data_3 = torch.FloatTensor(train_data_3).to(args.device)
    test_data_3 = torch.FloatTensor(test_data_3).to(args.device)

    train_label = torch.LongTensor(train_label).to(args.device)
    test_label = torch.LongTensor(test_label).to(args.device)

    # Build datasets
    train_dataset = MultiFunctionalConnectivityDataset(train_data_1, train_data_2, train_data_3, train_label)
    test_dataset = MultiFunctionalConnectivityDataset(test_data_1, test_data_2, test_data_3, test_label)

    # DataLoaders
    set_seed(args.seed)
    batch_size = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Compute class weights for loss balancing
    total = train_weight_index[0] + train_weight_index[1]
    weight = torch.tensor(
        [train_weight_index[1] / total, train_weight_index[0] / total], dtype=torch.float32
    ).to(args.device)

    return train_loader, test_loader, weight
