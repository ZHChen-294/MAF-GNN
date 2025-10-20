import os
import time
import numpy as np
import pandas as pd
from scipy import io


def make_fc_map(f_dir, f_names, f_out_dir, atlas="AAL"):
    """
    Compute functional connectivity (FC) matrices from ROI time series and save them as .mat files.
    Each FC matrix is Fisher Z-transformed.

    Parameters
    ----------
    f_dir : str
        Directory containing input .mat files (each with key 'ROISignals').
    f_names : list of str
        List of file names to process.
    f_out_dir : str
        Output directory for saving FC matrices.
    atlas : str, optional
        Atlas type for ROI selection. Options: {'AAL', 'Harvard', 'Craddock'}.
    """

    # --- Prepare output directories ---
    os.makedirs(f_out_dir, exist_ok=True)
    data_save_dir = os.path.join(f_out_dir, f"Data_{atlas}_FC_data")
    os.makedirs(data_save_dir, exist_ok=True)

    print(f"Processing atlas: {atlas}")
    print(f"Saving FC matrices to: {data_save_dir}")

    # --- Loop over all subjects ---
    for i, f_name in enumerate(f_names):
        print(f"[{i+1}/{len(f_names)}] Generating FC for: {f_name}")
        f_out_path = os.path.join(data_save_dir, f_name)

        # --- Load .mat file ---
        mat_path = os.path.join(f_dir, f_name)
        if not os.path.exists(mat_path):
            print(f"⚠️ File not found: {mat_path}")
            continue

        mat_data = io.loadmat(mat_path)
        if "ROISignals" not in mat_data:
            print(f"⚠️ 'ROISignals' key not found in {f_name}")
            continue

        roi_signals = mat_data["ROISignals"]  # shape: [timepoints, ROIs]

        # --- ROI selection by atlas ---
        if atlas == "AAL":
            roi_signals = roi_signals[:, :116]       # AAL: 116 ROIs
        elif atlas == "Harvard":
            roi_signals = roi_signals[:, 116:228]    # Harvard-Oxford: 112 ROIs
        elif atlas == "Craddock":
            roi_signals = roi_signals[:, 228:428]    # Craddock: 200 ROIs
        else:
            print(f"⚠️ Unknown atlas '{atlas}', using full signal matrix.")
            roi_signals = roi_signals

        # --- Compute Pearson correlation ---
        roi_df = pd.DataFrame(roi_signals)
        corr_df = roi_df.corr(method="pearson")

        # --- Fisher Z-transformation ---
        epsilon = 1e-5
        corr_matrix = np.clip(corr_df, -1 + epsilon, 1 - epsilon)
        fisher_z_matrix = 0.5 * np.log((1 + corr_matrix) / (1 - corr_matrix))

        # --- Save output (.mat) ---
        corr_matrix_dict = {"ROI_Functional_connectivity": fisher_z_matrix.to_numpy(dtype=np.float32)}
        io.savemat(f_out_path, corr_matrix_dict)

    print(f"✅ All FC maps generated and saved under: {data_save_dir}")


if __name__ == "__main__":
    # --- Path configuration ---
    f_dir = r""
    f_out_dir = r""
    f_names = [f for f in os.listdir(f_dir) if f.endswith(".mat")]

    atlas = "AAL"  # Options: 'AAL', 'Harvard', 'Craddock'

    start_time = time.time()
    make_fc_map(f_dir, f_names, f_out_dir, atlas=atlas)
    print(f"\n⏱️ Total processing time: {time.time() - start_time:.2f} seconds")
