# app.py
import os
import io
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn.functional as F
from torch import nn
from torch.serialization import add_safe_globals, safe_globals

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# =========================
# Model (same as training)
# =========================
class GCN(nn.Module):
    def __init__(self, num_node_features: int):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)          # graph-level pooling
        out = torch.sigmoid(self.fc(x)).view(-1)
        return out

# --- Allow-list the classes that appear in your checkpoint pickles
add_safe_globals([GCN, GCNConv])

# =========================
# Utilities
# =========================
def load_checkpoint_any(model_path: str, allow_unsafe: bool, map_location="cpu"):
    """
    Load ANY checkpoint format robustly.

    1) Try weights_only=True with safe globals allow-list (safe path)
    2) If that fails and allow_unsafe=True, try weights_only=False (UNSAFE)
    Returns: a state dict (preferred) or an nn.Module, as saved.
    """
    # 1) Safe path first
    try:
        with safe_globals([GCN, GCNConv]):
            obj = torch.load(model_path, map_location=map_location, weights_only=True)
        return obj
    except Exception as e_safe:
        if not allow_unsafe:
            raise RuntimeError(
                f"Safe load failed and unsafe load is disabled.\nSafe error: {e_safe}"
            ) from e_safe

    # 2) UNSAFE fallback (only if user allowed)
    try:
        obj = torch.load(model_path, map_location=map_location, weights_only=False)
        return obj
    except Exception as e_unsafe:
        raise RuntimeError(
            f"Unsafe load also failed.\nUnsafe error: {e_unsafe}"
        ) from e_unsafe


def extract_state_dict(ckpt_obj):
    """
    Normalize various checkpoint shapes into a plain state_dict.
    """
    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict()
    if isinstance(ckpt_obj, dict):
        # lightning- / custom-style wrappers
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        # plain state dict
        # (heuristic: values are tensors or contain obvious param keys)
        some_val = next(iter(ckpt_obj.values())) if len(ckpt_obj) else None
        if isinstance(some_val, torch.Tensor) or some_val is None:
            return ckpt_obj
    raise ValueError("Unsupported checkpoint format; cannot find a state_dict.")


def expected_in_feats_from_ckpt(model_path: str, allow_unsafe: bool) -> int:
    """
    Reads conv1.lin.weight from the checkpoint's state_dict to infer input feature count.
    """
    obj = load_checkpoint_any(model_path, allow_unsafe=allow_unsafe, map_location="cpu")
    sd = extract_state_dict(obj)
    w = sd.get("conv1.lin.weight", None)
    if w is None:
        # Some versions of PyG name underlying weight differently; try a few guesses
        # (GCNConv wraps Linear as .lin)
        # If not found, enumerate keys for debugging
        keys = "\n".join(sd.keys())
        raise KeyError("conv1.lin.weight not found in checkpoint state_dict.\nKeys:\n" + keys)
    return w.shape[1]


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, num_features: int, device: str, allow_unsafe: bool) -> nn.Module:
    """
    Instantiate model with 'num_features' and load checkpoint weights.
    """
    model = GCN(num_node_features=num_features).to(device)
    obj = load_checkpoint_any(model_path, allow_unsafe=allow_unsafe, map_location=device)
    # Try to load into the fresh instance
    if isinstance(obj, nn.Module):
        model.load_state_dict(obj.state_dict(), strict=False)
    else:
        sd = extract_state_dict(obj)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def parse_node_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def align_features(num_df: pd.DataFrame, expected_k: int, warn=True) -> pd.DataFrame:
    """
    Trim or zero-pad numeric columns to match expected_k.
    """
    k = num_df.shape[1]
    if k == expected_k:
        return num_df
    if k > expected_k:
        if warn:
            st.warning(f"Trimming features: found {k}, expected {expected_k}. Keeping first {expected_k}.")
        return num_df.iloc[:, :expected_k].copy()
    need = expected_k - k
    if warn:
        st.warning(f"Padding features: found {k}, expected {expected_k}. Adding {need} zero columns.")
    aligned = num_df.copy()
    for i in range(need):
        aligned[f"pad_{i}"] = 0.0
    return aligned


def parse_edge_index(df: pd.DataFrame, num_nodes: int) -> torch.Tensor:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        raise ValueError("edge_index.csv must have at least two numeric columns (src, dst).")
    edges = num_df.iloc[:, :2].astype(int).values
    if (edges < 0).any() or (edges >= num_nodes).any():
        raise ValueError("edge_index contains node indices out of range.")
    return torch.tensor(edges.T, dtype=torch.long)


def data_from_frames(x_df: pd.DataFrame, edge_df: pd.DataFrame):
    x = torch.tensor(x_df.values, dtype=torch.float32)
    n = x.shape[0]
    edge_index = parse_edge_index(edge_df, num_nodes=n)
    batch = torch.zeros(n, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="GCN Defect Prediction", layout="wide")
st.title("GCN Defect Prediction — Streamlit App")

with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Model checkpoint path", value="gcn_full_model.pth")
    device_opt = st.selectbox("Device", ["cpu", "cuda"], index=0)
    allow_unsafe = st.toggle(
        "Allow UNSAFE fallback load (weights_only=False) — use only if you trust the checkpoint",
        value=False,
        help="If safe load fails, the app will try unsafe load which can execute code embedded in the checkpoint."
    )
    use_uploaded = st.toggle("Use uploaded CSVs (node_features.csv, edge_index.csv)", value=False)
    st.markdown("---")
    st.caption("• node_features.csv: rows = nodes, columns = numeric features\n"
               "• edge_index.csv: 2 columns [src, dst] (0-based)")

exp_col1, exp_col2 = st.columns([2, 3])
with exp_col1:
    expected_k = None
    if os.path.exists(model_path):
        try:
            expected_k = expected_in_feats_from_ckpt(model_path, allow_unsafe=allow_unsafe)
            st.success(f"Checkpoint expects **{expected_k}** input features.")
        except Exception as e:
            st.error(f"Failed to read expected features from checkpoint:\n{e}")
    else:
        st.error(f"Model file not found: {model_path}")

# Data inputs
node_df_raw = None
edge_df = None

left, right = st.columns(2)
with left:
    st.subheader("Node Features")
    if use_uploaded:
        f_nodes = st.file_uploader("Upload node_features.csv", type=["csv"], key="nodes")
        if f_nodes is not None:
            try:
                node_df_raw = pd.read_csv(f_nodes)
                st.success(f"Loaded node_features.csv with shape {node_df_raw.shape}")
            except Exception as e:
                st.error(f"Could not read uploaded node_features.csv: {e}")
    else:
        if os.path.exists("node_features.csv"):
            try:
                node_df_raw = pd.read_csv("node_features.csv")
                st.info(f"Using local node_features.csv (shape {node_df_raw.shape})")
            except Exception as e:
                st.error(f"Could not read local node_features.csv: {e}")
        else:
            st.warning("node_features.csv not found locally.")

with right:
    st.subheader("Edge Index")
    if use_uploaded:
        f_edges = st.file_uploader("Upload edge_index.csv", type=["csv"], key="edges")
        if f_edges is not None:
            try:
                edge_df = pd.read_csv(f_edges)
                st.success(f"Loaded edge_index.csv with shape {edge_df.shape}")
            except Exception as e:
                st.error(f"Could not read uploaded edge_index.csv: {e}")
    else:
        if os.path.exists("edge_index.csv"):
            try:
                edge_df = pd.read_csv("edge_index.csv")
                st.info(f"Using local edge_index.csv (shape {edge_df.shape})")
            except Exception as e:
                st.error(f"Could not read local edge_index.csv: {e}")
        else:
            st.warning("edge_index.csv not found locally.")

st.markdown("---")

graph_ready = expected_k is not None and node_df_raw is not None and edge_df is not None

if graph_ready:
    try:
        num_df = parse_node_features(node_df_raw)
        feat_cols_before = num_df.columns.tolist()
        aligned_df = align_features(num_df, expected_k, warn=True)
        feat_cols_after = aligned_df.columns.tolist()

        # Offer download of the aligned CSV used for prediction
        csv_bytes = aligned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download aligned node_features.csv",
            data=csv_bytes,
            file_name="node_features_aligned.csv",
            mime="text/csv"
        )

        data = data_from_frames(aligned_df, edge_df)
        st.info(f"Graph ready: {data.num_nodes} nodes, {data.num_edges} edges, {aligned_df.shape[1]} features.")

        if feat_cols_before != feat_cols_after:
            with st.expander("Feature columns (before → after)"):
                n_before = len(feat_cols_before)
                n_after = len(feat_cols_after)
                n = max(n_before, n_after)
                before = feat_cols_before + [""] * (n - n_before)
                after = feat_cols_after + [""] * (n - n_after)
                st.dataframe(pd.DataFrame({"before": before, "after": after}), use_container_width=True)

    except Exception as e:
        st.error(f"Failed to build graph: {e}")
        graph_ready = False

pred_col, prob_col = st.columns(2)
if st.button("Run Prediction", disabled=not graph_ready):
    try:
        device = torch.device(device_opt if (device_opt == "cuda" and torch.cuda.is_available()) else "cpu")
        model = load_model(model_path, num_features=expected_k, device=str(device), allow_unsafe=allow_unsafe)
        with torch.no_grad():
            prob = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        p = float(prob.squeeze().cpu().item())
        y_hat = int(p >= 0.5)

        with pred_col:
            st.success(f"Predicted class: **{y_hat}** (thr=0.5)")
        with prob_col:
            st.metric("Defect probability", f"{p:.4f}")

        st.markdown("### Feature Summary (aligned)")
        feat_df = pd.DataFrame({
            "feature": feat_cols_after,
            "mean": data.x.cpu().numpy().mean(axis=0),
            "std":  data.x.cpu().numpy().std(axis=0)
        })
        st.dataframe(feat_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.expander("Troubleshooting"):
    st.markdown(
        """
- **Safe vs Unsafe load:** This app first tries a *safe* load (`weights_only=True`) with an allow-list for `GCN` and `GCNConv`.
  If it still fails and you **trust** the checkpoint, toggle **“Allow UNSAFE fallback load”** in the sidebar.
- **Feature count mismatch:** The app auto-trims/pads features to the expected count from the checkpoint.
- **Edge errors:** `edge_index.csv` must be 0-based node IDs within `[0, num_nodes-1]`.
- **GPU:** Select `cuda` only if a CUDA GPU is available.
        """
    )
