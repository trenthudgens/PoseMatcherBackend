from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

class Processor():

    def mse(self, imageA_path, imageB_path):
        '''
        calculates a very rough, pixel-wise similarity score
        between the two provided images.

        score is between 0-100.
        '''
        # Load the images
        imageA = cv2.imread(imageA_path)
        imageB = cv2.imread(imageB_path)

        # Convert images to grayscale
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # Resize images
        imageA = cv2.resize(imageA, (500, 500))
        imageB = cv2.resize(imageB, (500, 500))

        # Compute Mean Squared Error and Structural Similarity Index
        m = self.mse_calc(imageA, imageB)
        s = ssim(imageA, imageB)

        # Normalized, inverted MSE*100
        m = (1 - (m / 65536.0)) * 100

        # SSIM is in range -1 to 1. So, scale SSIM*50 + 50 to bring it in range 0 to 100.
        s = (s * 50) + 50

        # MSE and SSIM give similarity measures in the opposite direction, 
        # hence taking average of both the measures for final score
        return (m + s) / 2

    def mse_calc(self, imageA, imageB):
        # Calculating Mean Squared Error between two images
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err