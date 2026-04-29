# EdgeVec — Image Edge Detection using Vector Calculus Gradients
## 4th Semester Vector Calculus Project | Topic 4

---

## Vector Calculus Concepts Applied

| Concept | Formula | Application |
|---------|---------|-------------|
| Gradient | ∇I = (∂I/∂x, ∂I/∂y) | Detects direction of max intensity change |
| Gradient Magnitude | \|∇I\| = √(Gx² + Gy²) | Measures edge strength |
| Gradient Direction | θ = arctan(Gy / Gx) | Determines edge orientation |
| Laplacian | ∇²I = ∂²I/∂x² + ∂²I/∂y² | Second-order edge detection |

---

## Project Structure

```
edge_detection/
├── backend/
│   └── app.py          # Flask API — gradient computation
├── frontend/
│   └── index.html      # Web UI — interactive visualization
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the backend
```bash
cd backend
python app.py
# Server runs at http://localhost:5000
```

### 3. Open the frontend
Open `frontend/index.html` in your browser.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check server status |
| `/api/process` | POST | Process image, return gradient maps |
| `/api/demo` | GET | Generate a synthetic demo image |

### POST /api/process — Request Body
```json
{
  "image": "data:image/png;base64,...",
  "method": "sobel",       // sobel | prewitt | roberts
  "threshold": 80,         // 50–99 (percentile threshold)
  "laplacian": false       // include Laplacian output
}
```

### Response
```json
{
  "original":        "<base64 PNG>",
  "gradient_x":     "<base64 PNG>",   // ∂I/∂x
  "gradient_y":     "<base64 PNG>",   // ∂I/∂y
  "magnitude":      "<base64 PNG>",   // |∇I|
  "direction":      "<base64 PNG>",   // θ grayscale
  "direction_color":"<base64 PNG>",   // θ HSV color-coded
  "edges":          "<base64 PNG>",   // binary edge map
  "laplacian":      "<base64 PNG>",   // ∇²I (if requested)
  "stats": {
    "mean_magnitude": 12.4,
    "max_magnitude": 255.0,
    "edge_density_pct": 14.2,
    ...
  }
}
```

---

## How the Math Works

### Sobel Kernels (∂I/∂x and ∂I/∂y)
```
Gx kernel:          Gy kernel:
[-1  0  1]          [-1 -2 -1]
[-2  0  2]     vs   [ 0  0  0]
[-1  0  1]          [ 1  2  1]
```

These are applied via 2D convolution over the image.

### Gradient Magnitude
```python
magnitude = sqrt(Gx**2 + Gy**2)
```

### Edge Thresholding
```python
threshold = np.percentile(magnitude, threshold_pct)
edges = (magnitude >= threshold)
```

### Laplacian (∇²I)
```
Kernel:
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

---

## Results

The app outputs 6–7 visualization panels:
1. **Grayscale Input** I(x,y)
2. **Horizontal Gradient** ∂I/∂x
3. **Vertical Gradient** ∂I/∂y
4. **Gradient Magnitude** |∇I|
5. **Direction Map (color)** θ = arctan(Gy/Gx)
6. **Binary Edge Map** — final output
7. **Laplacian** ∇²I (optional)
