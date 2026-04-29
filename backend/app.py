"""
Image Edge Detection using Vector Calculus Gradients
Streamlit Version (Converted from Flask)

Vector Calculus Concepts Applied:
  - Gradient ∇f = (∂f/∂x, ∂f/∂y)
  - Gradient Magnitude |∇f| = sqrt((∂f/∂x)² + (∂f/∂y)²)
  - Gradient Direction θ = arctan(∂f/∂y / ∂f/∂x)
  - Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y²
"""

import streamlit as st
import numpy as np
from PIL import Image
import base64
import io

# ─────────────────────────────────────────
# KERNELS
# ─────────────────────────────────────────

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)

PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float64)

PREWITT_Y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float64)

LAPLACIAN = np.array([[ 0,  1,  0],
                      [ 1, -4,  1],
                      [ 0,  1,  0]], dtype=np.float64)

ROBERTS_X = np.array([[1, 0], [0, -1]], dtype=np.float64)
ROBERTS_Y = np.array([[0, 1], [-1, 0]], dtype=np.float64)

# ─────────────────────────────────────────
# CORE FUNCTIONS (UNCHANGED LOGIC)
# ─────────────────────────────────────────

def fast_convolve(image, kernel):
    from scipy.ndimage import convolve
    return convolve(image.astype(np.float64), kernel, mode='reflect')


def compute_gradient(gray, method='sobel'):
    if method == 'sobel':
        Gx = fast_convolve(gray, SOBEL_X)
        Gy = fast_convolve(gray, SOBEL_Y)
    elif method == 'prewitt':
        Gx = fast_convolve(gray, PREWITT_X)
        Gy = fast_convolve(gray, PREWITT_Y)
    elif method == 'roberts':
        Gx = fast_convolve(gray, np.pad(ROBERTS_X, ((0,1),(0,1))))[:-1, :-1]
        Gy = fast_convolve(gray, np.pad(ROBERTS_Y, ((0,1),(0,1))))[:-1, :-1]
        Gx = np.pad(Gx, ((0,1),(0,1)))
        Gy = np.pad(Gy, ((0,1),(0,1)))
    else:
        raise ValueError("Unknown method")

    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.degrees(np.arctan2(Gy, Gx))

    return Gx, Gy, magnitude, direction


def compute_laplacian(gray):
    return fast_convolve(gray, LAPLACIAN)


def normalize_to_uint8(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    return arr.astype(np.uint8)


def apply_threshold(magnitude, threshold_pct):
    threshold = np.percentile(magnitude, threshold_pct)
    return (magnitude >= threshold).astype(np.uint8) * 255


def compute_stats(Gx, Gy, magnitude, direction):
    return {
        "Mean Magnitude": float(np.mean(magnitude)),
        "Max Magnitude": float(np.max(magnitude)),
        "Std Magnitude": float(np.std(magnitude)),
        "Mean Gx": float(np.mean(np.abs(Gx))),
        "Mean Gy": float(np.mean(np.abs(Gy))),
        "Edge Density (%)": float(np.mean(magnitude > np.percentile(magnitude, 85)) * 100)
    }


def direction_colormap(direction, magnitude):
    hue = (direction + 180) / 360.0
    sat = np.ones_like(hue)
    val = np.clip(magnitude / (magnitude.max() + 1e-8), 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1)
    hsv_img = Image.fromarray((hsv * 255).astype(np.uint8), mode='HSV')
    return np.array(hsv_img.convert('RGB'))

# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────

st.set_page_config(page_title="Edge Detection", layout="wide")

st.title("🧠 Edge Detection using Vector Calculus")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

method = st.selectbox("Select Method", ["sobel", "prewitt", "roberts"])
threshold = st.slider("Threshold (%)", 0, 100, 80)
show_laplacian = st.checkbox("Show Laplacian")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # Resize
    max_dim = 512
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width*scale), int(img.height*scale)))

    gray = np.array(img.convert("L"), dtype=np.float64)

    Gx, Gy, mag, direction = compute_gradient(gray, method)

    edges = apply_threshold(mag, threshold)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original")
        st.image(normalize_to_uint8(Gx), caption="Gradient X")

    with col2:
        st.image(normalize_to_uint8(Gy), caption="Gradient Y")
        st.image(normalize_to_uint8(mag), caption="Magnitude")

    with col3:
        st.image(edges, caption="Edges")
        st.image(direction_colormap(direction, mag), caption="Direction Map")

    if show_laplacian:
        lap = compute_laplacian(gray)
        st.image(normalize_to_uint8(np.abs(lap)), caption="Laplacian")

    st.subheader("📊 Statistics")
    st.json(compute_stats(Gx, Gy, mag, direction))
