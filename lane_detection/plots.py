"""
plot.py - Image visualization utilities

A collection of functions for displaying images in various ways:
- Single images
- Multiple images in subplots
- Image comparisons
- Image grids
- With titles, colorbars, histograms, etc.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Union, Optional, Tuple, Dict

# Set default style
plt.style.use('default')

def show_image(image: np.ndarray, 
               title: str = None, 
               figsize: Tuple[int, int] = (10, 8),
               cmap: str = None,
               axis: bool = False,
               colorbar: bool = False,
               save_path: str = None,
               is_bgr: bool = True) -> None:
    """
    Display a single image
    
    Parameters:
    image: Input image (RGB, BGR, or grayscale)
    title: Title for the plot
    figsize: Figure size (width, height)
    cmap: Colormap (e.g., 'gray', 'viridis', 'jet')
    axis: Whether to show axes
    colorbar: Whether to show colorbar
    save_path: Path to save the figure
    is_bgr: Set to True if image is in BGR format (OpenCV default)
    """
    # Make a copy to avoid modifying original
    img_display = image.copy()
    
    # Convert BGR to RGB if needed
    if len(img_display.shape) == 3 and img_display.shape[2] == 3 and is_bgr:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    
    if len(img_display.shape) == 2:  # Grayscale
        plt.imshow(img_display, cmap=cmap or 'gray', interpolation='nearest')
    else:  # Color
        plt.imshow(img_display, interpolation='nearest')
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    
    if not axis:
        plt.axis('off')
    
    if colorbar:
        plt.colorbar(shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def show_images_grid(images: List[np.ndarray],
                     titles: List[str] = None,
                     cols: int = 3,
                     figsize: Tuple[int, int] = (15, 10),
                     cmap: str = None,
                     axis: bool = False,
                     save_path: str = None,
                     is_bgr: bool = True) -> None:
    """
    Display multiple images in a grid
    
    Parameters:
    images: List of images
    titles: List of titles for each image
    cols: Number of columns in the grid
    figsize: Figure size
    cmap: Colormap for grayscale images
    axis: Whether to show axes
    save_path: Path to save the figure
    is_bgr: Set to True if images are in BGR format (OpenCV default)
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    for i in range(n_images):
        img = images[i].copy()
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3 and is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if len(img.shape) == 2:  # Grayscale
            axes[i].imshow(img, cmap=cmap or 'gray', interpolation='nearest')
        else:  # Color
            axes[i].imshow(img, interpolation='nearest')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12)
        
        if not axis:
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def compare_images(images: List[np.ndarray],
                   titles: List[str] = None,
                   figsize: Tuple[int, int] = (15, 5),
                   cmap: str = None,
                   axis: bool = False,
                   show_diff: bool = False,
                   save_path: str = None,
                   is_bgr: bool = True) -> None:
    """
    Compare multiple images side by side (useful for before/after)
    
    Parameters:
    images: List of images to compare
    titles: Titles for each image
    figsize: Figure size
    cmap: Colormap
    axis: Whether to show axes
    show_diff: Show difference between first two images
    save_path: Path to save the figure
    is_bgr: Set to True if images are in BGR format (OpenCV default)
    """
    n_images = len(images)
    
    if show_diff and n_images >= 2:
        n_images += 1  # Add extra subplot for difference
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    for i in range(len(images)):
        img = images[i].copy()
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3 and is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap=cmap or 'gray', interpolation='nearest')
        else:
            axes[i].imshow(img, interpolation='nearest')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
        
        if not axis:
            axes[i].axis('off')
    
    # Show difference between first two images
    if show_diff and len(images) >= 2:
        diff_idx = len(images)
        
        # Ensure images are same type and size
        img1 = images[0].astype(np.float32)
        img2 = images[1].astype(np.float32)
        
        if len(img1.shape) == 3:
            diff = np.abs(img1 - img2).mean(axis=2)
        else:
            diff = np.abs(img1 - img2)
        
        axes[diff_idx].imshow(diff, cmap='hot', interpolation='nearest')
        axes[diff_idx].set_title('Difference', fontsize=14, fontweight='bold')
        axes[diff_idx].axis('off')
        
        # Add colorbar for difference
        plt.colorbar(axes[diff_idx].images[0], ax=axes[diff_idx], shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def show_with_histogram(image: np.ndarray,
                        title: str = None,
                        figsize: Tuple[int, int] = (12, 5),
                        bins: int = 256,
                        save_path: str = None,
                        is_bgr: bool = True) -> None:
    """
    Display image alongside its histogram
    
    Parameters:
    image: Input image
    title: Title for the plot
    figsize: Figure size
    bins: Number of histogram bins
    save_path: Path to save the figure
    is_bgr: Set to True if image is in BGR format (OpenCV default)
    """
    # Make a copy for display
    display_img = image.copy()
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3 and is_bgr:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Show image
    if len(image.shape) == 2:
        ax1.imshow(display_img, cmap='gray', interpolation='nearest')
    else:
        ax1.imshow(display_img, interpolation='nearest')
    
    ax1.set_title('Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Show histogram
    if len(image.shape) == 2:  # Grayscale
        ax2.hist(image.ravel(), bins=bins, range=[0, 256], 
                 color='black', alpha=0.7, histtype='stepfilled')
        ax2.set_xlim([0, 255])
        ax2.set_title('Grayscale Histogram', fontsize=12, fontweight='bold')
    else:  # Color
        colors = ('red', 'green', 'blue')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            ax2.plot(hist, color=color, alpha=0.8, linewidth=2, label=color.capitalize())
        ax2.set_xlim([0, 255])
        ax2.set_title('RGB Histogram', fontsize=12, fontweight='bold')
        ax2.legend()
    
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def show_image_pyramid(images: List[np.ndarray],
                       titles: List[str] = None,
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: str = None,
                       is_bgr: bool = True) -> None:
    """
    Display image pyramid (different scales)
    
    Parameters:
    images: List of images at different scales
    titles: Titles for each scale
    figsize: Figure size
    save_path: Path to save the figure
    is_bgr: Set to True if images are in BGR format (OpenCV default)
    """
    n_levels = len(images)
    
    fig, axes = plt.subplots(1, n_levels, figsize=figsize)
    if n_levels == 1:
        axes = [axes]
    
    for i, img in enumerate(images):
        display_img = img.copy()
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3 and is_bgr:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if len(img.shape) == 2:
            axes[i].imshow(display_img, cmap='gray', interpolation='nearest')
        else:
            axes[i].imshow(display_img, interpolation='nearest')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        else:
            axes[i].set_title(f'Level {i+1}', fontsize=12)
        
        axes[i].axis('off')
        
        # Add size annotation
        h, w = img.shape[:2]
        axes[i].text(0.5, -0.1, f'{w}×{h}', transform=axes[i].transAxes,
                    ha='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def show_image_channels(image: np.ndarray,
                       title: str = None,
                       figsize: Tuple[int, int] = (15, 4),
                       save_path: str = None,
                       is_bgr: bool = True) -> None:
    """
    Display individual channels of an RGB image
    
    Parameters:
    image: Input RGB image
    title: Title for the plot
    figsize: Figure size
    save_path: Path to save the figure
    is_bgr: Set to True if image is in BGR format (OpenCV default)
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Error: Image must be 3-channel")
        return
    
    # Convert to RGB for display if needed
    display_img = image.copy()
    if is_bgr:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original
    axes[0].imshow(display_img)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # For channel extraction, work with the original format
    if is_bgr:
        # BGR order
        channels = [
            ('Blue', (255,0,0)), 
            ('Green', (0,255,0)), 
            ('Red', (0,0,255))
        ]
        for idx, (name, color) in enumerate(channels):
            channel_img = np.zeros_like(display_img)
            channel_img[:, :, 2-idx] = display_img[:, :, 2-idx]  # Map to RGB for display
            axes[idx+1].imshow(channel_img)
            axes[idx+1].set_title(f'{name} Channel', fontsize=12, fontweight='bold')
            axes[idx+1].axis('off')
    else:
        # RGB order
        colors = ['Red', 'Green', 'Blue']
        rgb_indices = [0, 1, 2]
        for idx, (name, rgb_idx) in enumerate(zip(colors, rgb_indices)):
            channel_img = np.zeros_like(display_img)
            channel_img[:, :, rgb_idx] = display_img[:, :, rgb_idx]
            axes[idx+1].imshow(channel_img)
            axes[idx+1].set_title(f'{name} Channel', fontsize=12, fontweight='bold')
            axes[idx+1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def create_comparison_grid(images_dict: Dict[str, np.ndarray],
                          cols: int = 3,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: str = None,
                          is_bgr: bool = True) -> None:
    """
    Display a grid of images with labels from dictionary
    
    Parameters:
    images_dict: Dictionary with keys as titles and values as images
    cols: Number of columns
    figsize: Figure size
    save_path: Path to save the figure
    is_bgr: Set to True if images are in BGR format (OpenCV default)
    """
    titles = list(images_dict.keys())
    images = list(images_dict.values())
    
    show_images_grid(images, titles, cols, figsize, save_path=save_path, is_bgr=is_bgr)


def show_image_with_colorbar(image: np.ndarray,
                            title: str = None,
                            cmap: str = 'viridis',
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: str = None,
                            is_bgr: bool = True) -> None:
    """
    Display image with a colorbar (useful for heatmaps, distance transforms, etc.)
    
    Parameters:
    image: Input image (usually grayscale with meaningful values)
    title: Title
    cmap: Colormap
    figsize: Figure size
    save_path: Path to save
    is_bgr: Set to True if image is in BGR format (only relevant for color images)
    """
    display_img = image.copy()
    
    # Convert BGR to RGB if it's a color image
    if len(image.shape) == 3 and image.shape[2] == 3 and is_bgr:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    
    if len(display_img.shape) == 2:
        im = plt.imshow(display_img, cmap=cmap, interpolation='nearest')
    else:
        im = plt.imshow(display_img, interpolation='nearest')
    
    plt.colorbar(im, shrink=0.8)
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create a test image
    img = np.random.rand(100, 100) * 255
    img = img.astype(np.uint8)
    
    # Test different functions
    print("Testing plot.py functions...")
    
    # Show single image
    show_image(img, title="Test Image")
    
    # Show with histogram
    show_with_histogram(img, title="Image with Histogram")
    
    # Show grid
    images = [img, img, img, img]
    titles = ["Image 1", "Image 2", "Image 3", "Image 4"]
    show_images_grid(images, titles, cols=2)