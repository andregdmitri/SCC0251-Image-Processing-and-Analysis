"""
    Nome: Andre Guarnier De Mitri
    nUSP: 11395579
    Disciplina: SCC0251
    Ano: 2023-01
    Turma: 2023101
    Assignment 01: Enhancement and Superresolution
"""

import numpy as np
import imageio

def histogram(Li, no_levels = 256):
    """
    Creates a histogram of the image

    :param np.ndarray Li: Image in the np.ndarray format
    :param int no_levels: Number of levels in the grayscale [0:255]
    :return: A histogram of the image
    :rtype: np.ndarray
    """
    hist = np.zeros(no_levels).astype(int)
    for i in range(no_levels):
        pixels_value_i = np.sum(Li == i)
        hist[i] = pixels_value_i
    return(hist)

def histogram_equalization(Li, hist, no_levels, size):
    """
    Equalizes a given histogram (hist).

    :param np.ndarray Li: Image in the np.ndarray format
    :param np.ndarray hist: Cumulative histogram of one or all the images
    :param int no_levels: Number of levels in the grayscale [0:255]
    :param int size: size of the resulting image used in the equalization
    :return: Equalized image according to the histogram
    :rtype: np.ndarray
    """
    histC = np.zeros(no_levels)
    histC[0] = hist[0] # first value (intensity 0)
    for i in range(1,  no_levels):
        histC[i] = hist[i] + histC[i-1]
    hist_transform = np.zeros(no_levels)
    N, M = Li.shape
    Li_eq = np.zeros([N,M])
    for z in range(no_levels):
        s = ((no_levels-1)/float(size*size))*histC[z]
        Li_eq[ np.where(Li == z) ] = s
    return (Li_eq)


def histogram_single(L, H):
    """
    Equalizes each of the 4 low quality images, each with their own histogram, and stores them inside the list

    :param list L: A list containing all the images in the np.ndarray format
    :param np.ndarray H: The original high resolution image
    :return: List with all the equalized images
    :rtype: list
    """
    N, M = L[0].shape
    Li_chapeu = np.zeros((256, 256))
    L_chapeu = []
    hist = np.zeros(256).astype(int)
    for i in range(4):
        hist = histogram(L[i])
        Li_chapeu = histogram_equalization(L[i], hist, 256, 256)
        L_chapeu.append(Li_chapeu)
    return L_chapeu

def histogram_joint(L, H):
    """
    Creates a single cumulative histogram using all the 4 low quality images and transforms them

    :param list L: A list containing all the images in the np.ndarray format
    :param np.ndarray H: The original high resolution image
    :return: List with all the equalized images
    :rtype: list
    """
    N, M = L[0].shape
    Li_chapeu = np.zeros((256, 256))
    L_chapeu = []
    hist = np.zeros(256).astype(int)
    for i in range(4):
        hist += histogram(L[i])
    for i in range(4):
        Li_chapeu = histogram_equalization(L[i], hist, 256, 512)
        L_chapeu.append(Li_chapeu)  
    return L_chapeu

def gammaCorrection(L, H, gamma):
    """
    Pixel-wise enhancement Gamma Correction

    :param list L: A list containing all the images in the np.ndarray format
    :param np.ndarray H: The original high resolution image
    :param float gamma: The value of the gamma adjustment
    :return: List with all the images after the enhacement
    :rtype: list
    """
    N, M = L[0].shape
    Li_chapeu = np.zeros((256, 256))
    L_chapeu = []
    for i in range(4):
        for x in range(N):
            for y in range(M):
                Li_chapeu[x, y] = (255*np.power(L[i][x, y]/255.0, 1/gamma)).astype(np.uint)
        L_chapeu.append(Li_chapeu)
    return L_chapeu

def superresolution(L):
    """
    Creates a single image from 4 low quality images

    :param list L: A list containing all the images in the np.ndarray format
    :return: All the images merged together
    :rtype: np.ndarray
    """
    N, M = L[0].shape
    H_chapeu = np.zeros((2*N, 2*M))
    for i in range(N*2):
        for j in range (M*2):
            if((i%2 == 0) and (j%2 == 0)): # top left
                H_chapeu[i, j] = L[0][int(i/2), int(j/2)]
            elif((i%2 == 0) and (j%2 == 1)): # top right
                H_chapeu[i, j] = L[1][int(i/2), int(j/2)]
            elif((i%2 == 1) and (j%2 == 0)): # bottom left
                H_chapeu[i, j] = L[2][int(i/2), int(j/2)]
            else: #bottom right
                H_chapeu[i, j] = L[3][int(i/2), int(j/2)]
    return H_chapeu

def RMSE(H, H_chapeu):
    """
    Compares the RMSE (root mean squared error) of the original with the new image

    :param np.ndarray H: The original image
    :param np.ndarray H_chapeu: The new image after the image processing method
    :return: RMSE (root mean squared error)
    :rtype: float
    """
    rmse = np.float64(0)
    N, M = H_chapeu.shape
    for i in range(N):
        for j in range(M):
            rmse += np.power(H[i, j] - H_chapeu[i, j], 2)/(N*N)
    return np.sqrt(rmse)

if __name__ == '__main__':
    # User input
    imglow = input().rstrip()
    imghigh = input().rstrip()
    F = input().rstrip() 
    gamma = float(input())

    # Using imageio
    L = []
    for i in range(4):
        L.append(imageio.v3.imread(f'{imglow}{i}.png'))
    H = imageio.v3.imread(imghigh)

    # Operations
    if (F == '0'):
        H_chapeu = superresolution(L)
    elif (F == '1'):
        H_chapeu = histogram_single(L, H)
        H_chapeu = superresolution(H_chapeu)
    elif (F == '2'):
        H_chapeu = histogram_joint(L, H)
        H_chapeu = superresolution(H_chapeu)
    elif (F == '3'):
        H_chapeu = gammaCorrection(L, H, gamma)
        H_chapeu = superresolution(H_chapeu)
    else:
        raise TypeError("Invalid Operation")
    print(f'{RMSE(H, H_chapeu):.4f}')