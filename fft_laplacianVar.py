import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_fft(img_original):

    img = cv2.imread(img_original, 0) #read as greyscale
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f) #shift the sero-frequency component to the center of the spectrum
    magnitude_spectrum = 20*np.log(np.abs(f_shift))


    # rows, cols = img.shape
    # crow, ccol = int(rows/2) , int(cols/2)
    #
    # mask = np.zeros((rows,cols),np.uint8)
    # mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # f2_shift = f_shift * mask
    # f2_ishift = np.fft.ifftshift(f2_shift)
    # f2_inverse = np.abs(np.fft.ifft2(f2_ishift))


    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum,cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

def get_image_var(imgPath):
    image = cv2.imread(imgPath,0)
    image_var = cv2.Laplacian(image, cv2.CV_64F).var() #calculate the variation of picture after laplacian (core: cv2.CV_64F)

    return image_var


image_fft('img.jpg')
image_var = get_image_var('img.jpg')
print(image_var)


