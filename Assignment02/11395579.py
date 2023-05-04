"""
    Nome: Andre Guarnier De Mitri
    nUSP: 11395579
    Disciplina: SCC0251
    Ano: 2023-01
    Turma: 2023101
    Assignment 02: Fourier Transform
"""
import numpy as np
import imageio
from numpy.fft import ifft2, ifftshift

#----------------------- FILTER DEFINITIONS ---------------------
'''
    These are the Functions responsible for creating the filters
'''
def ideal_filter(D):
    P, Q = D.shape
    for u in range(P):
        for v in range(Q):
            D[u,v] = np.sqrt((u-(P/2))**2 + (v-(Q/2))**2)
    return(D)

def Lowpass(D, r):
    P, Q = D.shape
    H = np.zeros((P,Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            if (D[u,v] <= r):
                H[u,v] = 1
            else:
                H[u,v] = 0
    return(H)

def Highpass(D, r):
    P, Q = D.shape
    H = np.zeros((P,Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            if (D[u,v] <= r):
                H[u,v] = 0
            else:
                H[u,v] = 1
    return(H)

def Laplacian(H):
    P, Q = H.shape
    for u in range(P):
        for v in range(Q):
            H[u,v] = -4*(np.pi**2)*((u-(P/2))**2 + (v-(Q/2))**2)
    return H

def Gaussian(H, sigma1, sigma2):
    P, Q = H.shape
    for u in range(P):
        for v in range(Q):
            x = ((u-(P/2))**2)/(2*((sigma1)**2)) + ((v-(Q/2))**2)/(2*((sigma2)**2))
            H[u,v] = np.exp(-x)
    return H

def low_Butterworth(H, D0, n):
    P, Q = H.shape
    D = ideal_filter(H)
    for u in range(P):
        for v in range(Q):
            H[u,v] = 1/(1+((D[u,v]/D0)**(2*n)))
    return H

def high_Butterworth(H, D0, n):
    P, Q = H.shape
    D = ideal_filter(H)
    for u in range(P):
        for v in range(Q):
            H[u,v] = 1/(1+((D[u,v]/D0)**(-2*n)))
    return H
#-----------------------------------------------


######################### OPERATIONS###################
def ideal_low_pass(r):
    """
    Creates the Ideal Low Pass Filter

    :param float r: Inner circle width (distance from the origin)
    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.zeros((2*P,2*Q), dtype=np.float32)
    H_ideal = ideal_filter(H)
    H_ideal_low = Lowpass(H_ideal, r)
    return(H_ideal_low)

def ideal_high_pass(r):
    """
    Creates the Ideal High Pass Filter

    :param float r: Inner circle width (distance from the origin)
    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.zeros((2*P,2*Q), dtype=np.float32)
    H_ideal = ideal_filter(H)
    H_ideal_high = Highpass(H_ideal, r)
    return(H_ideal_high)

def ideal_band_pass(r1, r2):
    """
    Creates the Ideal Band Pass Filter

    :param float r1: Outer circle width (distance from the origin)
    :param float r2: Inner circle width (distance from the origin)
    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.zeros((2*P,2*Q), dtype=np.float32)
    H_ideal = ideal_filter(H)
    H_ideal_low1 = Lowpass(H_ideal, r1)
    H_ideal_low2 = Lowpass(H_ideal, r2)
    if(r1 > r2):
        H_ideal_band = H_ideal_low1 - H_ideal_low2
    else:
        H_ideal_band = H_ideal_low2 - H_ideal_low1
    return(H_ideal_band)

def laplacian_high_pass():
    """
    Creates the Laplacian High Pass Filter

    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.zeros((2*P,2*Q), dtype=np.float32)
    H_laplacian = 255-Laplacian(H)
    return(H_laplacian)

def gaussian_low_pass(sigma1, sigma2):
    """
    Creates the Gaussian Low Pass Filter

    :param float sigma1: width of the Gaussian curve
    :param float sigma2: how sharp the filter is goign to be
    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.zeros((2*P,2*Q), dtype=np.float32)
    H_gaussian = Gaussian(H, sigma1, sigma2)
    return(H_gaussian)

def butterworth_low_pass(D0, n):
    """
    Creates the butterwoth low pass filter

    :param float D0: Distance from origin
    :param float n: Order of the transfer function
    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.zeros((2*P,2*Q), dtype=np.float32)
    butterworth_low = low_Butterworth(H, D0, n)
    return(butterworth_low)

def butterworth_high_pass(D0, n):
    """
    Creates the butterwoth high pass filter

    :param float D0: Distance from origin
    :param float n: Order of the transfer function
    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.ones((2*P,2*Q), dtype=np.float32)
    butterworth_high = high_Butterworth(H, D0, n)
    return(butterworth_high)

def butterworth_band_reject(D0, D1, n0, n1):
    """
    Creates the butterwoth band reject filter

    :param float D0: Distance from origin (outer circle)
    :param float n0: Order of the transfer function
    :param float D1: Distance from origin (inner circle)
    :param float n1: Order of the transfer function
    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.ones((2*P,2*Q), dtype=np.float32)
    H_high = butterworth_high_pass(D0, n0)
    H_low = butterworth_low_pass(D1, n1)
    H_band_rej = H_high + H_low
    return(H_band_rej)

def butterworth_band_pass(D0, D1, n0, n1):
    """
    Creates the butterwoth band pass filter

    :param float D0: Distance from origin (outer circle)
    :param float n0: Order of the transfer function
    :param float D1: Distance from origin (inner circle)
    :param float n1: Order of the transfer function
    :return: H_chapeu, the filter image
    :rtype: np.ndarray
    """
    H = np.ones((2*P,2*Q), dtype=np.float32)
    H_low1 = butterworth_low_pass(D0, n0)
    H_low2 = butterworth_low_pass(D1, n1)
    if(D0 > D1):
        H_band = H_low1 - H_low2
    else:
        H_band = H_low2 - H_low1
    return(H_band)
#############################################################################

def RMSE(H, H_chapeu):
    """
    Compares the RMSE (root mean squared error) of the original with the new image

    :param np.ndarray H: The original image
    :param np.ndarray H_chapeu: The new image after the image processing method
    :return: RMSE (root mean squared error)
    :rtype: float
    """
    rmse = np.float32(0)
    N, M = H_chapeu.shape
    for i in range(N):
        for j in range(M):
            rmse += np.power(H[i, j] - H_chapeu[i, j], 2)/(N*M)
    return np.sqrt(rmse)


#********************* MAIN *********************
if __name__ == '__main__':
    # User input
    I = imageio.v3.imread(input().rstrip())
    H = imageio.v3.imread(input().rstrip())
    i = np.uint(input())
    if i > 9:
        raise TypeError("Invalid index")
    
    # 2D FFT of the input image
    P, Q = I.shape
    F = np.fft.fftshift(np.fft.fft2(I))

    # Operations
    if (i == 0):
        r = float(input())
        filter = ideal_low_pass(r)
    elif (i == 1):
        r = float(input())
        filter = ideal_high_pass(r)
    elif (i == 2):
        r1 = float(input())
        r2 = float(input())
        filter = ideal_band_pass(r1, r2)
    elif (i == 3):
        filter = laplacian_high_pass()
    elif (i == 4):
        sigma1 = float(input())
        sigma2 = float(input())
        filter = gaussian_low_pass(sigma1, sigma2)
    elif (i == 5):
        D0 = float(input())
        n = float(input())
        filter = butterworth_low_pass(D0, n)
    elif (i == 6):
        D0 = float(input())
        n = float(input())
        filter = butterworth_high_pass(D0, n)
    elif (i == 7):
        D0 = float(input())
        n1 = float(input())
        D1 = float(input())
        n2 = float(input())
        filter = butterworth_band_reject(D0, n1, D1, n2)
    elif (i == 8):
        D0 = float(input())
        n1 = float(input())
        D1 = float(input())
        n2 = float(input())
        filter = butterworth_band_pass(D0, n1, D1, n2)
    else:
        raise TypeError("Invalid Operation")
    
    # Resulting images
    x1_filter = int((F.shape[0])*0.5)
    x2_filter = int((F.shape[0])*1.5)
    y1_filter = int((F.shape[1])*0.5)
    y2_filter = int((F.shape[1])*1.5)

    filter = filter[x1_filter:x2_filter, y1_filter:y2_filter]
    F_hat = F * filter # Applying the Filter to F
    H_hat_img = ifft2(ifftshift(F_hat)) #Bringing back to the image subspace
    H_hat = np.real(H_hat_img) # The real part of the imaginary number
    H_hat = 255*((H_hat-np.min(H_hat))/(np.max(H_hat)-np.min(H_hat))) # Normalizing
    print(f'{RMSE(H, H_hat):.4f}')