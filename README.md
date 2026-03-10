
# Image Processing & Lane Detection

This repository contains two distinct computer vision projects built with Python and OpenCV. The first project applies a series of classic image processing filters to create cartoon-style images. The second project implements a Hough Transform pipeline from scratch to detect road lane lines.

## Projects Overview

### 1. 🖼️ Image Cartoonifying
Transforms real-world images into cartoon-like images by applying various image processing techniques. The goal is to make flat regions more uniform and edges more distinct, creating a comic book effect.

#### Cartoonifying Pipeline
The process follows a three-step approach, as described in [Task 1: Image Cartoonifying Using Image Processing Filters](#task-1-image-cartoonifying-using-image-processing-filters).

**Step 1: Black & White Sketch Generation**
- Convert image to grayscale.
- Apply a **Median filter** for noise reduction while preserving edges (`apply_laplacian.py`).
- Detect edges using a **Laplacian filter** (`apply_laplacian.py`).
- Apply **binary thresholding** to create clean, bold sketch lines (`apply_laplacian.py`).

**Step 2: Color Painting Generation**
- Apply an edge-preserving smoothing filter (Bilateral Filter) to smooth flat regions while maintaining edge sharpness, creating a poster-like effect (`apply_bilateral.py`).
- Performance is optimized by potentially working at a lower resolution or applying multiple small bilateral filters.

**Step 3: Cartoon Creation**
- Overlay the binary sketch (edge mask) onto the color painting.
- The final output combines smooth colored regions with bold black sketch lines (`overlay.py`).

### 2. 🛣️ Lane Detection (Hough Transform)
Detects road lane lines in an image using a classical computer vision pipeline, with a custom implementation of the Hough Transform.

#### Lane Detection Pipeline
The end-to-end pipeline (`hough_transform.py`) performs the following steps:
1.  **Load Image**: Reads the input image.
2.  **Smoothing**: Applies a median filter to reduce noise.
3.  **Edge Detection**: Uses the Canny edge detector to find strong edges.
4.  **Region of Interest (ROI) Masking**: Masks out everything except the expected road area in front of the car.
5.  **Hough Accumulation**: A custom function builds the Hough accumulator array by voting in (ρ, θ)-space for each edge pixel.
6.  **Plot Accumulator**: Visualizes the accumulator as a heat-map.
7.  **Non-Maximum Suppression**: Suppresses non-peak values in the accumulator to isolate true lines.
8.  **Peak Extraction**: Extracts the most prominent peaks from the accumulator, representing the detected lines.
9.  **Draw Lines**: Draws the detected lines onto the original image and displays all intermediate and final results.

## File Structure

```
.
├── image_processing_cartoonifying/   # Files for the cartoonifying project
│   ├── apply_bilateral.py            # Applies bilateral filter for color painting
│   ├── apply_laplacian.py            # Applies Laplacian for edge detection & sketch
│   ├── overlay.py                     # Overlays sketch on painting to create cartoon
│   └── plots.py                        # Utility functions for plotting images
├── lane_detection/                    # Files for the lane detection project
│   └── hough_transform.py             # Custom Hough Transform pipeline
├── README.md                           # This file
└── .vscode/                             # VS Code configuration (optional)
```

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/o-ronin-o/image_processing-lane_detection.git
    cd image_processing-lane_detection
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install opencv-python numpy matplotlib scipy
    ```

## Usage

### Cartoonifying an Image

Navigate to the `image_processing_cartoonifying` directory and run the scripts sequentially, or integrate the functions into your own script.

1.  **Generate Sketch**: Edit and run `apply_laplacian.py`. This will create a binary edge mask (e.g., `binary.png`).
    ```bash
    python apply_laplacian.py
    ```
2.  **Generate Color Painting**: Edit and run `apply_bilateral.py` to smooth the image.
    ```bash
    python apply_bilateral.py
    ```
3.  **Create Final Cartoon**: Run `overlay.py` to combine the sketch and the painting.
    ```bash
    python overlay.py
    ```

### Lane Detection

1.  **Place your image**: Ensure you have a test image named `road.png` in the `lane_detection` directory, or modify the `image_path` in the script.
2.  **Run the pipeline**
    ```bash
    cd ../lane_detection   # Adjust path as needed
    python hough_transform.py
    ```
    The script will display the pipeline steps and save the final result as `lane_detection_output.png`.

## Key Parameters

### Cartoonifying
- **Median filter size**: Controls noise reduction.
- **Laplacian filter**: Used for edge detection.
- **Binary threshold**: Converts edges to black/white sketch lines (in `apply_laplacian.py`).
- **Bilateral filter parameters**: Color strength, positional strength, and filter size (in `apply_bilateral.py`).

### Lane Detection
All parameters can be tuned in the `lane_detection_pipeline` function call in `hough_transform.py`.
- `canny_low` & `canny_high`: Thresholds for the Canny edge detector.
- `num_peaks`: Number of lines to detect.
- `hough_threshold`: Minimum votes for a line to be considered a peak.
- `line_color` & `line_thickness`: Visual properties of the drawn lines.

## Expected Outputs

### Cartoonifying
- **Sketch**: A black and white line drawing.
- **Painting**: A smooth, poster-like color image.
- **Cartoon**: The final result with bold sketch lines overlaid on the smooth painting.

### Lane Detection
The pipeline will output a series of plots showing:
- Original, smoothed, Canny edges, and ROI-masked images.
- The Hough accumulator as a heatmap.
- The final image with detected lane lines overlaid in green.

## Dependencies
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- SciPy

