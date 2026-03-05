import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from plots import show_image

def demonstrate_bilateral_parameters(image):
    """
    Show effect of different bilateral filter parameters
    """
    # Convert BGR to RGB once at the start
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    filtered = cv2.bilateralFilter(image_rgb, d=9, sigmaColor=75, sigmaSpace=150)

    return filtered

  
if __name__ == "__main__":
    # Read image
    img = cv2.imread('image.png', cv2.IMREAD_COLOR)
    
    # Check if image exists
    if img is None:
        print("Error: image.png not found!")
    else:
        img = demonstrate_bilateral_parameters(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = demonstrate_bilateral_parameters(img)
        show_image(img)
        cv2.imwrite('smoothed.png', img)

