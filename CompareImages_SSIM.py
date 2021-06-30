# Reference: https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
# Based on: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-

from skimage.metrics import structural_similarity
import argparse
#import imutils
import cv2

# Load the two input images
#imageA = cv2.imread('C:\\Git\\Python\\Images\\Family.jpg')                     # Original Image
#imageB = cv2.imread('C:\\Git\\Python\\Images\\Family(smaller).jpg')            # Smaller Image of Original
#imageB = cv2.imread('C:\\Git\\Python\\Images\\Madelyn.jpg')                    # Different picture
#imageB = cv2.imread('C:\\Git\\Python\\Images\\Family.jpg')                     # Duplicate of Original Image

# Abby Johnson
imageA = cv2.imread('C:\\TEMP\\ComparePhotoTest\\O365Photos\\Abby Johnson.jpg') # Original Image
imageB = cv2.imread('C:\\TEMP\\ComparePhotoTest\\Photos processed\\3178.jpg')   # O365 Avatar

# Matthew Reule
#imageA = cv2.imread('C:\\TEMP\\ComparePhotoTest\\O365Photos\\Matthew Reule.jpg') # Original Image
#imageB = cv2.imread('C:\\TEMP\\ComparePhotoTest\\Photos processed\\3139.jpg') # O365 Avatar

# James Walters
#imageA = cv2.imread('C:\\TEMP\\ComparePhotoTest\\O365Photos\\James Walters.jpg') # Original Image
#imageB = cv2.imread('C:\\TEMP\\ComparePhotoTest\\Photos processed\\3080.jpg') # O365 Avatar

#imageB = cv2.imread('C:\\TEMP\\ComparePhotoTest\\Photos processed\\1z83yn.jpg') # Picture of Picard TNG


# Resize ImageA to be same size as ImageB
height = imageB.shape[0]
width = imageB.shape[1]
dim = (width, height)

resized_imageA = cv2.resize(imageA, dim, interpolation = cv2.INTER_AREA)

# Convert the images to grayscale
grayA = cv2.cvtColor(resized_imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
(score, diff) = structural_similarity(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# Print only the score
print("SSIM: {}".format(score))





