import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from plots import show_image, compare_images

def smooth_image(image, size=5):
    """
    Reduce noise using a 2-D median smoothing filter.
    """
    if len(image.shape) == 3:                       
        smoothed = np.stack(
            [median_filter(image[:, :, c], size=size) for c in range(image.shape[2])],
            axis=2
        ).astype(image.dtype)
    else:                                           
        smoothed = median_filter(image, size=size).astype(image.dtype)
    return smoothed

def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Detect edges using Canny's algorithm.
    High threshold values are used to suppress most noise.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def region_of_interest(edges, vertices=None):
    """
    Mask the edge image so that only the road region is kept.
    """
    h, w = edges.shape[:2]

    if vertices is None:
        # Default trapezoid covering the lower half of the frame
        vertices = np.array([[
            (0,          h),               # bottom-left
            (w * 0.45,   h * 0.6),         # top-left
            (w * 0.55,   h * 0.6),         # top-right
            (w,          h),               # bottom-right
        ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges


def hough_accumulate(edges, theta_res=1, rho_res=1):
    """
    Build the Hough accumulator array by voting in (ρ, θ)-space.

    Algorithm
    ---------
    For each edge point (x, y):
        For θ = 0 … 180:
            ρ = x·cos θ + y·sin θ
            H(θ, ρ) += 1
    """
    h, w = edges.shape
    diag = int(np.ceil(np.hypot(h, w)))     # max possible ρ

    thetas = np.deg2rad(np.arange(0, 180, theta_res))
    rhos   = np.arange(-diag, diag + 1, rho_res)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int64)

    # Edge pixel coordinates
    ys, xs = np.nonzero(edges)

    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t
        rho_idxs = np.round((rho_vals - rhos[0]) / rho_res).astype(int)

        # Clip to valid range and vote
        valid = (rho_idxs >= 0) & (rho_idxs < len(rhos))
        accumulator[rho_idxs[valid], np.where(valid)[0]] += 1

    return accumulator, thetas, rhos

def non_max_suppression(accumulator, neighborhood=10):
    """
    Suppress non-peak values in the accumulator.

    For each local maximum: any cell within *neighborhood* cells in both
    dimensions that has a lower count is zeroed, keeping only the true peaks.
    """
    suppressed = accumulator.copy()
    nr, nt = accumulator.shape

    # Sort indices by descending vote count
    flat_indices = np.argsort(accumulator.ravel())[::-1]

    for flat_idx in flat_indices:
        r_idx, t_idx = np.unravel_index(flat_idx, accumulator.shape)
        if suppressed[r_idx, t_idx] == 0:
            continue                                    # already suppressed
        r_lo = max(0,  r_idx - neighborhood)
        r_hi = min(nr, r_idx + neighborhood + 1)
        t_lo = max(0,  t_idx - neighborhood)
        t_hi = min(nt, t_idx + neighborhood + 1)

        window = suppressed[r_lo:r_hi, t_lo:t_hi]
        peak   = window.max()
        if suppressed[r_idx, t_idx] == peak:
            window[window < peak] = 0                   # suppress neighbours
        else:
            suppressed[r_idx, t_idx] = 0               # this cell is not max

    return suppressed


def find_peaks(accumulator, thetas, rhos, threshold=None, num_peaks=10):
    """
    Extract the strongest (ρ, θ) peaks from the (suppressed) accumulator.
    """
    if threshold is None:
        threshold = int(accumulator.max() * 0.5)

    peaks = []
    acc = accumulator.copy()

    for _ in range(num_peaks):
        idx   = np.unravel_index(np.argmax(acc), acc.shape)
        votes = acc[idx]
        if votes < threshold:
            break
        rho   = rhos[idx[0]]
        theta = thetas[idx[1]]
        peaks.append((rho, theta, votes))
        acc[idx] = 0                # zero the peak so next argmax finds next one

    return peaks

def draw_lines(image, peaks, color=(0, 255, 0), thickness=3):
    """
    Draw Hough lines on a copy of the image.
    """
    output = image.copy()
    h, w   = image.shape[:2]

    for rho, theta, _ in peaks:
        a = np.cos(theta)
        b = np.sin(theta)

        # Convert (ρ, θ) → two far-apart points
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)

        cv2.line(output, (x1, y1), (x2, y2), color, thickness)

    return output


def plot_accumulator(accumulator, thetas, rhos, title="Hough Accumulator (ρ-θ space)"):
    """
    Display the Hough accumulator array as a heat-map.

    Parameters
    ----------
    accumulator : ndarray  Vote array.
    thetas      : ndarray  Angle values (radians) – used as x-axis labels.
    rhos        : ndarray  ρ values – used as y-axis labels.
    title       : str      Plot title.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(
        accumulator,
        aspect='auto',
        cmap='hot',
        extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]]
    )
    plt.colorbar(label='Votes')
    plt.xlabel('θ (degrees)')
    plt.ylabel('ρ (pixels)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def lane_detection_pipeline(image_path, roi_vertices=None,
                             canny_low=50, canny_high=150,
                             hough_threshold=None, num_peaks=10,
                             line_color=(0, 255, 0), line_thickness=3):
    """
    End-to-end road lane detection pipeline.

    Steps
    -----
    1. Load image
    2. Median smoothing
    3. Canny edge detection
    4. ROI masking
    5. Hough transform accumulation
    6. Plot accumulator
    7. Non-maximum suppression
    8. Peak extraction
    9. Draw lines & display all results
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── 2. Smooth ─────────────────────────────────────────────────────────────
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    smoothed  = smooth_image(gray, size=5)

    # ── 3. Canny edge detection ───────────────────────────────────────────────
    edges = detect_edges(smoothed, canny_low, canny_high)

    # ── 4. ROI masking ────────────────────────────────────────────────────────
    roi_edges = region_of_interest(edges, roi_vertices)

    # ── 5. Hough accumulation ─────────────────────────────────────────────────
    print("Running Hough accumulation … (this may take a moment)")
    accumulator, thetas, rhos = hough_accumulate(roi_edges)

    # ── 6. Plot accumulator ───────────────────────────────────────────────────
    plot_accumulator(accumulator, thetas, rhos)

    # ── 7. Non-maximum suppression ────────────────────────────────────────────
    suppressed = non_max_suppression(accumulator, neighborhood=10)

    # ── 8. Peak extraction ────────────────────────────────────────────────────
    peaks = find_peaks(suppressed, thetas, rhos,
                       threshold=hough_threshold, num_peaks=num_peaks)
    print(f"Detected {len(peaks)} line(s):")
    for i, (r, t, v) in enumerate(peaks):
        print(f"  Line {i+1}: ρ={r:.1f} px, θ={np.rad2deg(t):.1f}°, votes={v}")

    # ── 9. Draw & display ─────────────────────────────────────────────────────
    result_bgr = draw_lines(img_bgr, peaks, color=line_color,
                            thickness=line_thickness)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # Display pipeline results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    titles_imgs = [
        ("Original Image",        img_rgb,     None),
        ("Smoothed (Median 5×5)", smoothed,    'gray'),
        ("Canny Edges",           edges,        'gray'),
        ("ROI Edge Mask",         roi_edges,   'gray'),
        ("Suppressed Accumulator",
         suppressed.astype(np.float32) / max(suppressed.max(), 1), 'hot'),
        ("Lane Detection Result", result_rgb,  None),
    ]
    for ax, (title, im, cmap) in zip(axes.flat, titles_imgs):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Road Lane Detection – Hough Transform Pipeline",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return result_bgr

if __name__ == "__main__":
    result = lane_detection_pipeline(
        image_path="road.png",      
        canny_low=50,
        canny_high=150,
        num_peaks=10,
    )
    cv2.imwrite("lane_detection_output.png", result)
    print("Saved result to lane_detection_output.png")
