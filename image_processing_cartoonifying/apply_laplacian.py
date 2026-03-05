import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.ndimage import median_filter


import sys

# if len(sys.argv) > 1:
#     STRENGTH = float(sys.argv[1])
#     print(STRENGTH)
    
# else:
#     STRENGTH = 2
#     print(STRENGTH)




def binary_threshold(image,thres = 55,max_val =255):
    ret, binary = cv2.threshold(laplacian, thres, max_val, cv2.THRESH_BINARY)

    return binary    

def edge_detection(image):
    """
    Sharpen image using Laplacian filter
    
    Parameters:
    image: Input image
    """
    
    
    image = median_filter(image, size=5) # to deal with noise 
    
    laplacian  = cv2.Laplacian(image, cv2.CV_64F, ksize = 3, delta = 20)

    edges = cv2.convertScaleAbs(laplacian)


    return edges

if __name__ == "__main__":

    img_gray_direct = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)


    # Method 1: Read color image and convert to grayscale
    img_color = cv2.imread('image.png', cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray_from_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)


    # Method 2: Read directly as grayscale
    img_gray_direct = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)


    # Apply sharpening to grayscale images
    laplacian = edge_detection(img_gray_direct)

    # Display results
    plt.figure(figsize=(15,5))

    plt.subplot(131)
    plt.imshow(img_rgb)
    plt.title('Original Color')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(img_gray_direct, cmap='gray')
    plt.title('Grayscale (Direct Read)')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    
    
    binary = binary_threshold(laplacian)

    
    plt.imshow(binary, cmap='gray')
    plt.title(f'Binary Threshold (T={55})')
    plt.axis('off')

    
    cv2.imwrite('binary.png', binary)

    # Print image information
    print(f"Original color image shape: {img_color.shape}")
    print(f"Grayscale image shape: {img_gray_direct.shape}")
    print(f"Grayscale image data type: {img_gray_direct.dtype}")
    print(f"Grayscale pixel range: {img_gray_direct.min()} - {img_gray_direct.max()}")