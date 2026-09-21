"""
Streamlit app for cephalometric landmark detection (CephNet).

Run locally:
    streamlit run app.py

Deployment notes are in README.md.
"""

import io
import os

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed landmark order used during training — must match the checkpoint
LM_NAMES = [
    "Po", "Go", "R", "Co", "UMT", "PNS", "UPM", "Or",
    "LMT", "Ar", "LIA", "UIA", "LPM", "ANS", "Pog`",
    "B", "Li", "N`", "N", "Pog", "LIT", "A", "Ls",
    "Sn", "S", "Gn", "Me", "UIT", "Pn",
]
N_LM = len(LM_NAMES)

# Path to the checkpoint inside the deployed app.
# Keep the .pth file next to app.py (see README for hosting large files).
CHECKPOINT_PATH = os.environ.get("CEPHNET_CKPT", "checkpoint_v3_multiscale.pth")


# --------------------------------------------------------------------------
# Model definition (unchanged from training/inference code)
# --------------------------------------------------------------------------

class SoftArgmaxHead(nn.Module):
    def __init__(self, in_ch, n_lm=N_LM, temperature=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_lm, kernel_size=1)
        self.temperature = temperature

    def forward(self, feat):
        heat = self.conv(feat)
        B, N, H, W = heat.shape

        heat = heat.flatten(2)
        heat = F.softmax(heat / self.temperature, dim=-1)
        heat = heat.view(B, N, H, W)

        ys = torch.linspace(0, 1, H, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(0, 1, W, device=feat.device, dtype=feat.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        x = (heat * gx).sum(dim=(-1, -2))
        y = (heat * gy).sum(dim=(-1, -2))

        return torch.stack([x, y], dim=-1)


class CephNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=False):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            features_only=True,
            out_indices=(1, 2, 3),
        )

        channels = self.backbone.feature_info.channels()
        c8, c16, c32 = channels

        self.proj8 = nn.Sequential(nn.Conv2d(c8, 128, 1), nn.BatchNorm2d(128), nn.GELU())
        self.proj16 = nn.Sequential(nn.Conv2d(c16, 128, 1), nn.BatchNorm2d(128), nn.GELU())
        self.proj32 = nn.Sequential(nn.Conv2d(c32, 128, 1), nn.BatchNorm2d(128), nn.GELU())

        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.head = SoftArgmaxHead(in_ch=128, n_lm=N_LM, temperature=0.1)

    def forward(self, x):
        f8, f16, f32 = self.backbone(x)
        f8 = self.proj8(f8)

        f16 = self.proj16(f16)
        f16 = F.interpolate(f16, size=f8.shape[-2:], mode="bilinear", align_corners=False)

        f32 = self.proj32(f32)
        f32 = F.interpolate(f32, size=f8.shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([f8, f16, f32], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)


# --------------------------------------------------------------------------
# Cached model loading — runs once per session, not on every rerun
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CephNet checkpoint…")
def load_model(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return None, None

    model = CephNet(pretrained=False).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    meta = {
        "epoch": checkpoint.get("epoch", "n/a"),
        "best_sdr2": checkpoint.get("best_sdr2", "n/a"),
    }
    return model, meta


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def predict_landmarks(model, img_gray: np.ndarray):
    """img_gray: single-channel uint8 numpy array (H, W)."""
    H, W = img_gray.shape

    resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(resized).float() / 255.0
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)[0]

    pred = pred.cpu().numpy()
    pred[:, 0] *= W
    pred[:, 1] *= H
    return pred


def render_plot(img_gray: np.ndarray, landmarks: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img_gray, cmap="gray")
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=30, c="red")

    for name, (x, y) in zip(LM_NAMES, landmarks):
        ax.text(x + 5, y + 5, name, fontsize=8, color="yellow")

    ax.set_xlim(0, img_gray.shape[1])
    ax.set_ylim(img_gray.shape[0], 0)
    ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Cephalometric Landmark Detection", layout="wide")
st.title("Cephalometric Landmark Detection")
st.caption(f"Device: {DEVICE} · Landmarks: {N_LM} · Input size: {IMG_SIZE}px")

with st.sidebar:
    st.header("Model")
    ckpt_upload = st.file_uploader(
        "Checkpoint (.pth) — optional, only if not already bundled with the app",
        type=["pth", "pt"],
    )
    if ckpt_upload is not None:
        # Save uploaded checkpoint to disk so torch.load can read it, then bust the cache
        tmp_ckpt_path = "uploaded_checkpoint.pth"
        with open(tmp_ckpt_path, "wb") as f:
            f.write(ckpt_upload.getbuffer())
        checkpoint_path = tmp_ckpt_path
    else:
        checkpoint_path = CHECKPOINT_PATH

    model, meta = load_model(checkpoint_path)

    if model is None:
        st.error(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Upload a .pth file above, or bundle one with the app (see README)."
        )
    else:
        st.success("Model loaded")
        st.write(f"Epoch: {meta['epoch']}")
        st.write(f"Best SDR@2.0: {meta['best_sdr2']}")

st.subheader("Upload a cephalometric X-ray")
image_file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if image_file is not None:
    file_bytes = np.frombuffer(image_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Could not decode this image file.")
    elif model is None:
        st.warning("Load a model checkpoint first (see sidebar).")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Input", use_container_width=True, clamp=True)

        with st.spinner("Running inference…"):
            landmarks = predict_landmarks(model, img)

        with col2:
            fig = render_plot(img, landmarks)
            st.pyplot(fig, use_container_width=True)

        st.subheader("Predicted coordinates")
        st.dataframe(
            {
                "Landmark": LM_NAMES,
                "X (px)": [round(float(x), 1) for x, y in landmarks],
                "Y (px)": [round(float(y), 1) for x, y in landmarks],
            },
            use_container_width=True,
        )

        # Download predictions as CSV
        import pandas as pd

        df = pd.DataFrame(
            {
                "landmark": LM_NAMES,
                "x_px": landmarks[:, 0],
                "y_px": landmarks[:, 1],
            }
        )
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download landmarks as CSV",
            data=csv_buf.getvalue(),
            file_name=f"{os.path.splitext(image_file.name)[0]}_landmarks.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a cephalometric radiograph (JPG/PNG) to run landmark detection.")
