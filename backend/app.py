"""
Image Edge Detection using Vector Calculus Gradients
Backend: Flask API
Vector Calculus Concepts Applied:
  - Gradient ∇f = (∂f/∂x, ∂f/∂y) — spatial rate of change of intensity
  - Gradient Magnitude |∇f| = sqrt((∂f/∂x)² + (∂f/∂y)²)
  - Gradient Direction θ = arctan(∂f/∂y / ∂f/∂x)
  - Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² — divergence of the gradient
"""

from flask import Flask, request, jsonify, after_this_request
import numpy as np
from PIL import Image
import base64
import io
import json

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options(path):
    return jsonify({}), 200

# ─────────────────────────────────────────
# VECTOR CALCULUS KERNEL DEFINITIONS
# ─────────────────────────────────────────

# Sobel kernels — approximate partial derivatives ∂f/∂x and ∂f/∂y
SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)   # ∂I/∂x

SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float64)   # ∂I/∂y

# Prewitt kernels — simpler gradient approximation
PREWITT_X = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float64)

PREWITT_Y = np.array([[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]], dtype=np.float64)

# Laplacian kernel — ∇²f = divergence of gradient
LAPLACIAN = np.array([[ 0,  1,  0],
                       [ 1, -4,  1],
                       [ 0,  1,  0]], dtype=np.float64)

# Roberts Cross — diagonal gradient approximation
ROBERTS_X = np.array([[1, 0], [0, -1]], dtype=np.float64)
ROBERTS_Y = np.array([[0, 1], [-1, 0]], dtype=np.float64)


def convolve2d(image, kernel):
    """Manual 2D convolution — applies kernel to compute spatial derivatives."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='reflect')
    result = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result


def fast_convolve(image, kernel):
    """Faster convolution using stride tricks / vectorized numpy."""
    from scipy.ndimage import convolve
    return convolve(image.astype(np.float64), kernel, mode='reflect')


def compute_gradient(gray, method='sobel'):
    """
    Compute the gradient vector field ∇I of the grayscale image.
    Returns Gx, Gy, magnitude, direction, stats.
    """
    if method == 'sobel':
        Gx = fast_convolve(gray, SOBEL_X)    # ∂I/∂x
        Gy = fast_convolve(gray, SOBEL_Y)    # ∂I/∂y
    elif method == 'prewitt':
        Gx = fast_convolve(gray, PREWITT_X)
        Gy = fast_convolve(gray, PREWITT_Y)
    elif method == 'roberts':
        gray_small = gray[:-1, :-1]
        Gx = fast_convolve(gray, np.pad(ROBERTS_X, ((0,1),(0,1))))[:-1, :-1]
        Gy = fast_convolve(gray, np.pad(ROBERTS_Y, ((0,1),(0,1))))[:-1, :-1]
        # Pad back to original size
        Gx = np.pad(Gx, ((0,1),(0,1)))
        Gy = np.pad(Gy, ((0,1),(0,1)))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Gradient magnitude: |∇I| = sqrt(Gx² + Gy²)
    magnitude = np.sqrt(Gx**2 + Gy**2)

    # Gradient direction: θ = arctan(Gy / Gx) — in degrees
    direction = np.degrees(np.arctan2(Gy, Gx))

    return Gx, Gy, magnitude, direction


def compute_laplacian(gray):
    """
    Laplacian ∇²I = ∂²I/∂x² + ∂²I/∂y²
    Measures divergence of the gradient (second-order edges).
    """
    lap = fast_convolve(gray, LAPLACIAN)
    return lap


def normalize_to_uint8(arr):
    """Normalize array to [0, 255] uint8."""
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    return arr.astype(np.uint8)


def array_to_base64(arr):
    """Convert numpy array to base64 PNG string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def apply_threshold(magnitude, threshold_pct):
    """Apply threshold to gradient magnitude — keep only strong edges."""
    threshold = np.percentile(magnitude, threshold_pct)
    binary = (magnitude >= threshold).astype(np.uint8) * 255
    return binary


def compute_stats(Gx, Gy, magnitude, direction):
    """Compute vector calculus statistics for display."""
    return {
        "mean_magnitude": float(np.mean(magnitude)),
        "max_magnitude": float(np.max(magnitude)),
        "std_magnitude": float(np.std(magnitude)),
        "mean_Gx": float(np.mean(np.abs(Gx))),
        "mean_Gy": float(np.mean(np.abs(Gy))),
        "dominant_direction_deg": float(np.mean(direction[magnitude > np.percentile(magnitude, 75)])),
        "edge_density_pct": float(np.mean(magnitude > np.percentile(magnitude, 85)) * 100),
    }


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Edge Detection API running"})


@app.route('/api/process', methods=['POST'])
def process_image():
    """
    Main endpoint: accepts base64 image + parameters,
    returns gradient analysis results.
    """
    data = request.get_json()
    img_b64 = data.get('image')
    method = data.get('method', 'sobel')
    threshold_pct = float(data.get('threshold', 80))
    show_laplacian = data.get('laplacian', False)

    # Decode image
    img_bytes = base64.b64decode(img_b64.split(',')[-1])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Resize for performance if too large
    max_dim = 512
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # Convert to grayscale numpy array
    gray = np.array(img.convert('L'), dtype=np.float64)

    # ─── VECTOR CALCULUS: Compute Gradient Field ───
    Gx, Gy, magnitude, direction = compute_gradient(gray, method)

    # Normalize outputs
    Gx_norm = normalize_to_uint8(Gx)
    Gy_norm = normalize_to_uint8(Gy)
    mag_norm = normalize_to_uint8(magnitude)
    dir_norm = normalize_to_uint8(direction)

    # Edge map via thresholding
    edges = apply_threshold(magnitude, threshold_pct)

    # Colorize direction map (HSV-like)
    dir_color = direction_colormap(direction, magnitude)

    results = {
        "original": array_to_base64(np.array(img.convert('L'))),
        "gradient_x": array_to_base64(Gx_norm),
        "gradient_y": array_to_base64(Gy_norm),
        "magnitude": array_to_base64(mag_norm),
        "direction": array_to_base64(dir_norm),
        "direction_color": array_to_base64(dir_color),
        "edges": array_to_base64(edges),
        "stats": compute_stats(Gx, Gy, magnitude, direction),
        "method": method,
        "image_size": list(gray.shape),
    }

    if show_laplacian:
        lap = compute_laplacian(gray)
        lap_norm = normalize_to_uint8(np.abs(lap))
        results["laplacian"] = array_to_base64(lap_norm)

    return jsonify(results)


def direction_colormap(direction, magnitude):
    """
    Map gradient direction (angle) → color using HSV.
    Angle [−180°, 180°] → Hue [0, 1]
    Magnitude normalized → Value
    """
    hue = (direction + 180) / 360.0  # normalize to [0,1]
    sat = np.ones_like(hue)
    mag_n = magnitude / (magnitude.max() + 1e-8)
    val = np.clip(mag_n * 3, 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    from PIL import Image as PILImage
    hsv_img = PILImage.fromarray((hsv * 255).astype(np.uint8), mode='HSV')
    rgb_img = hsv_img.convert('RGB')
    return np.array(rgb_img)


@app.route('/api/demo', methods=['GET'])
def demo_image():
    """Generate a synthetic test image for demonstration."""
    size = 256
    img = np.zeros((size, size), dtype=np.float64)

    # Add shapes with sharp edges
    img[60:100, 60:100] = 200    # Square
    for i in range(size):        # Circle
        for j in range(size):
            if (i-170)**2 + (j-170)**2 < 40**2:
                img[i, j] = 180
    # Diagonal line
    for k in range(size):
        if 0 <= k < size:
            img[k, k] = 220
    # Gradient region
    img[10:50, 150:220] = np.linspace(50, 200, 70)

    # Add slight noise
    img += np.random.normal(0, 5, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    b64 = array_to_base64(img)
    return jsonify({"image": f"data:image/png;base64,{b64}"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
