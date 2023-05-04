"""
    Nome: Andre Guarnier De Mitri
    nUSP: 11395579
    Disciplina: SCC0251
    Ano: 2023-01
    Turma: 2023101
    Assignment 03: Image Descriptors
"""
import numpy as np
import imageio
from scipy.ndimage import convolve

def luminance(X):
    """
    Transforms a list of RGB images into a list of Gray-Scale images

    :param np.ndarray X: A list of images with RGB
    :return np.ndarray X_new: A list of images with in Gray Scale
    """
    X_new = [] # The new image list
    for image in X: # For each image of the X list
        image = np.array(image, copy=True).astype(float)
        new_img = np.zeros((image.shape[0], image.shape[1]))
        new_img = np.floor(0.299*image[:, :, 0]+ 0.587*image[:, :, 1] + 0.114*image[:, :, 2])
        X_new.append(new_img)
    return X_new

def HoG_descriptor(X):
    """
    Histogram of Oriented Gradients, this descriptor is a good way of capturing how 
    the textures in an image are “arranged” by looking at the angle of the gradients.

    :param np.ndarray X: A list of images in Gray Scale
    :return np.ndarray dg: A list magnitutes accumulated in bins accoring to the gradient and the angle
    """
    Wsx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) # Gradient in X
    Wsy = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # Gradient in Y
    dg_list = [] # List of all dg calcultated from each image
    for image in X: # For each image of the X list
        image = np.array(image, copy=True).astype(float)
        gx = convolve(image, Wsx) # getting the gradient in x
        gy = convolve(image, Wsy) # getting the gradient in Y
        P, Q = gx.shape
        M = np.sqrt(gx**2 + gy**2)/(np.sqrt(gx**2 + gy**2).sum()) #Magnitute
        phi = np.ndarray([P, Q]) # Angle
        phi = np.arctan(gy/gx)
        phi = np.degrees(phi+(np.pi/2))
        phi_d = np.array(phi//20) # Creating the bins
        dg = np.zeros(9)
        for x in range(P):
            for y in range(Q):
                for i in range(9):
                    if(phi_d[x,y] == i): 
                        dg[i] += M[x,y] # Accumulating in the bins
        dg_list.append(dg)
    return dg_list

def euclidian_distance_list(dg_list_X, dg_list_Xtest):
    """
    Calculates the Euclidian Distance for all the images in the list

    :param np.ndarray dg_list_X: A list of the Trainning images processed by the HoG Descriptor
    :param np.ndarray dg_list_Xtest: A list of the Testing images processed by the HoG Descriptor
    :return np.ndarray dg: A list of the euclidian distance between the Trainning and Testing images after being processed by the HoG Descriptor
    """
    list_size = len(dg_list_X)
    D_X_list = []
    D_dga_dgb = np.zeros(list_size)
    for dg_a in dg_list_Xtest:
        i = 0
        for dg_b in dg_list_X:
            D_dga_dgb[i] = np.sqrt(np.sum((dg_a-dg_b)**2)) # Gets distance between 2 images
            i = i+1
        D_X_list.append(D_dga_dgb)
        D_dga_dgb = np.zeros(list_size) # Resetting the matrix
    return D_X_list

def KNN(dg_list_X1, dg_list_X2, dg_list_Xtest, k=3):
    """
    Calculates the Nearest Neighboor between the Testing images and the Trainning images

    :param np.ndarray dg_list_X1: A list of the Trainning images  without humans processed by the HoG Descriptor
    :param np.ndarray dg_list_X2: A list of the Trainning images with humans processed by the HoG Descriptor
    :param np.ndarray dg_list_Xtest: A list of the Testing images processed by the HoG Descriptor
    :return np.ndarray dg: A list classifying the each Test image as 1 (has humans) or 0 (doesnt have humans)
    """

    # Euclidian distance between Train and Test
    D_X1_list = euclidian_distance_list(dg_list_X1, dg_list_Xtest)
    D_X2_list = euclidian_distance_list(dg_list_X2, dg_list_Xtest)

    # Nearest Neighboor
    vote_list = [] # A list for the final result
    d_x_size = len(D_X1_list)
    for i in range(d_x_size):
        smallest_X1 = sorted(D_X1_list[i])[:k-1] # k-1 Smallest values in X1
        smallest_X2 = sorted(D_X2_list[i])[:k-1] # k-1 Smallest values in X2
        minimum = sorted(smallest_X1 + smallest_X2)[:3] # Smallest value by a list composed of both X1 and X2
        votes = 0
        for x in range(k):
            for y in range(k-1):
                if (minimum[x] == smallest_X1[y]): # If it belongs to the X1 class, it gets a vote
                    votes +=1
        if (votes >= k-1):
            vote_list.append(0) # If most of the votes belong to the X1 class, the class is 0
        else:
            vote_list.append(1) # otherwise, it belongs with X2 and the class is 1
    return vote_list

if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    X0 = [] # Images without humans
    X1 = [] # Images with humans
    Xtest = [] # Test images

    # Images directory (user INPUT)
    x0_dir = input().rstrip()
    x1_dir = input().rstrip()
    xtest_dir = input().rstrip()

    # Creating 3 lists, containing images
    for image in x0_dir.split(" "):
        X0.append(imageio.v3.imread(image))
    for image in x1_dir.split(" "):
        X1.append(imageio.v3.imread(image))
    for image in xtest_dir.split(" "):
        Xtest.append(imageio.v3.imread(image))
    
    # Transforming images to black&white using the Luminance technique
    X0_gray = luminance(X0)
    X1_gray = luminance(X1)
    Xtest_gray = luminance(Xtest)

    # Histogram of Oriented Gradients (HoG)
    X0_HoG = HoG_descriptor(X0_gray)
    X1_HoG = HoG_descriptor(X1_gray)
    Xtest_HoG = HoG_descriptor(Xtest_gray)

    # Output
    print(*KNN(X0_HoG, X1_HoG, Xtest_HoG, 3))