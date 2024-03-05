import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading and displaying original image
clrImg= cv2.imread("squares.jpg")
plt.imshow(clrImg), plt.title('Original Image')
plt.show()

# converting and displaying image to grayscale
img = cv2.cvtColor(clrImg,cv2.COLOR_BGR2GRAY)
plt.imshow(img,cmap=plt.cm.get_cmap('gray')), plt.title('Image in Grayscale')

# smoothing the image with sigma=1
img = cv2.GaussianBlur(img,(3,3),1)

# displaying smoothed image
plt.figure(figsize=(4, 4))
plt.imshow(img,cmap=plt.cm.get_cmap('gray')), plt.title('Smoothed Original Image')

# function to display images
def plot2Images(img1, img2, txt1, txt2):
    plt.figure(figsize=(60, 60))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(txt1,fontsize = 60)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(txt2,fontsize = 60)
    plt.axis('off')

# calculating Ix
def gradientX(img):
    kernelX = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    Ix = cv2.filter2D(src=img, ddepth=-1, kernel=kernelX)
    return Ix

# calculating Iy
def gradientY(img):
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Iy = cv2.filter2D(src=img, ddepth=-1, kernel=kernelY)
    return Iy

Ix = gradientX(img)
Iy = gradientY(img)

# displaying Ix and Iy
plt.figure(figsize=(10, 10))
plt.subplot(1,2,1), plt.imshow(Ix,cmap=plt.cm.get_cmap('gray')), plt.title('Ix')
plt.subplot(1,2,2), plt.imshow(Iy,cmap=plt.cm.get_cmap('gray')), plt.title('Iy')

# calculating structure tensor and smoothing the images with sigma=2.5 (greater than that on original image)
Ixx = cv2.GaussianBlur(Ix**2, (3,3), 2.5)
Iyy = cv2.GaussianBlur(Iy**2, (3,3), 2.5)
Ixy = cv2.GaussianBlur(Ix*Iy, (3,3), 2.5)

# function for corner detection
def cornerDetection(option):
    alpha = 0.06
    beta = 0.05
    detA = Ixx * Iyy - Ixy ** 2
    traceA = Ixx + Iyy
    lambda1 = lambda2 = np.zeros((Ixx.shape[0],Ixx.shape[1]))
    for i in range(Ixx.shape[0]):
        for j in range(Ixx.shape[1]):
            ST = [[Ixx[i][j],Ixy[i][j]],[Ixy[i][j],Iyy[i][j]]]    
            _, l, _ = np.linalg.svd(ST)
            lambda1[i][j] = l[0]
            lambda2[i][j] = l[1]

    # Harris and Stephens (1988) calculation
    if option == 1:
        R = detA - alpha * traceA ** 2
    
    # Shi and Tomasi (1994) calculation
    elif option == 2:
        R = lambda1
    
    # Rohr (1994) calculation
    elif option == 3:
        R = detA

    # Triggs (2004) calculation
    elif option == 4:
        R = lambda1 - beta * lambda2

    # Brown, Szeliski, and Winder (2005) calculation
    else:
        R = np.zeros((Ixx.shape[0],Ixx.shape[1]))
        for i in range(Ixx.shape[0]):
            for j in range(Ixx.shape[1]):
                if traceA[i][j] == 0:
                    R[i][j] = 0
                else:
                    R[i][j] = detA[i][j] / traceA[i][j]
    return R

option = int(input('Enter 1 for Harris and Stephens (1988) Corner Detection\n 2 for Shi and Tomasi (1994)\n 3 for Rohr (1994)\n 4 for Triggs (2004)\n 5 for Brown, Szeliski, and Winder (2005)\n'))
if (option < 1 or option > 5):
    print('Wrong Input.')
else: 
    R = cornerDetection(option)

# thresholding
threshold = 0.02
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i][j] < threshold:
            R[i][j] = 0

def non_max_suppression(harris_resp, window_size=3):
    # Initialize output as all zeros
    output = np.zeros_like(harris_resp)
    # Get image shape
    rows, cols = harris_resp.shape
    
    # Pad image with zeros to handle edge cases
    padded = np.pad(harris_resp, (window_size//2, window_size//2), mode='constant')
    
    # Iterate over every pixel in the response map
    for row in range(rows):
        for col in range(cols):
            # Extract window around the current pixel
            window = padded[row:row+window_size, col:col+window_size]
            # Check if current pixel is a local maximum within the window
            if harris_resp[row, col] >= np.max(window):
                output[row, col] = harris_resp[row, col]
    
    return output

# non maxima supression
R=non_max_suppression(R,3)

# detecting corners from R
imgCopy = np.copy(clrImg)

for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i][j] > 0:
            imgCopy[i , j] = [255, 0, 0]

# displaying input image and final output i.e., corners detected
plot2Images(clrImg, imgCopy, 'Original Image', 'Corners Detected')
