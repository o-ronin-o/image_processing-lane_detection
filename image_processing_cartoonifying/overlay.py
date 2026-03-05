import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sketch_overlay(painting, edge_mask, edge_color=(0, 0, 0)):
    """
    Overlay edge mask onto painting with black background
    
    Parameters:
    painting: Bilateral filtered image (color)
    edge_mask: Binary edge mask (white edges on black background)
    edge_color: Color for edges (default black for sketch effect)
    """
    # Create black background
    h, w = painting.shape[:2]
    result = np.zeros_like(painting)
    
    # Copy painting pixels where edge_mask is 0 (non-edges)
    # Assuming edge_mask has white edges (255) on black background (0)
    non_edge_mask = edge_mask == 0
    
    if len(painting.shape) == 3:  # Color image
        result[non_edge_mask] = painting[non_edge_mask]
        # Edges remain black (from the background)
    else:  # Grayscale
        result[non_edge_mask] = painting[non_edge_mask]
    
    return result

# Example usage
if __name__ == "__main__":
    # Read images
    painting = cv2.imread('smoothed.png')
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = cv2.imread('binary.png')
  
    
    # Create sketch overlay
    sketch = create_sketch_overlay(painting, edges)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(painting)
    plt.title('Original')
    plt.axis('off')
    
   
    
    plt.subplot(132)
    plt.imshow(sketch)
    plt.title('Sketch Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()