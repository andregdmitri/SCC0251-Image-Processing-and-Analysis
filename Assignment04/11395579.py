"""
    Nome: Andre Guarnier De Mitri
    nUSP: 11395579
    Disciplina: SCC0251
    Ano: 2023-01
    Turma: 2023101
    Assignment 04: Mathematical Morphology
"""

import numpy as np
import imageio

def flood_fill(I, x_seed, y_seed, c):
    """
        Given a seed pixel P, we get all the pixels of the same color 
        painting a specific region and detecting connected components within an image

        :param np.ndarray I: input image in the np.ndarray format
        :param int x_seed: X coordinate of a the seed pixel (where the algorithm starts)
        :param int y_seed: Y coordinate of a the seed pixel (where the algorithm starts)
        :return list connection_list: A list of all pixels connected with the seed pixel
    """
    neighbor_list = [] # A list of pixels in the neighboordhood of another pixel
    connection_list = [] # The list of all the connected pixels
    neighbor_list.append(tuple((x_seed, y_seed)))
    while(len(neighbor_list) != 0):
            i, j = neighbor_list[0] # Getting the first number of neighbor_list (starts with the seed)

            # given a seed pixel P, if the neighboorhgood of a this pixel is of the same color
            # AND if it is not in the list of Neighboors (neighbor_list)
            # AND if its not in the list of pixels that are already connected (connection_list)
            # then we add the pixel to the list of Neighboors (neighbor_list)
            if (I[i, j] == I[i+1, j] and ((tuple((i+1, j)) in connection_list) == False)\
                and ((tuple((i+1, j)) in neighbor_list) == False)):
                neighbor_list.append(tuple((i+1, j)))
            if (I[i, j] == I[i, j+1] and ((tuple((i, j+1)) in connection_list) == False)\
                and ((tuple((i, j+1)) in neighbor_list) == False)):
                neighbor_list.append(tuple((i, j+1)))
            if (I[i, j] == I[i-1, j] and ((tuple((i-1, j)) in connection_list) == False)\
                and ((tuple((i-1, j)) in neighbor_list) == False)):
                neighbor_list.append(tuple((i-1, j)))
            if (I[i, j] == I[i, j-1] and ((tuple((i, j-1)) in connection_list) == False)\
                and ((tuple((i, j-1)) in neighbor_list) == False)):
                neighbor_list.append(tuple((i, j-1)))
            if (c == 8):
                if (I[i, j] == I[i+1, j+1] and ((tuple((i+1, j+1)) in connection_list) == False)\
                and ((tuple((i+1, j+1)) in neighbor_list) == False)):
                    neighbor_list.append(tuple((i+1, j+1)))
                if (I[i, j] == I[i+1, j-1] and ((tuple((i+1, j-1)) in connection_list) == False)\
                and ((tuple((i+1, j-1)) in neighbor_list) == False)):
                    neighbor_list.append(tuple((i+1, j-1)))
                if (I[i, j] == I[i-1, j+1] and ((tuple((i-1, j+1)) in connection_list) == False)\
                and ((tuple((i-1, j+1)) in neighbor_list) == False)):
                    neighbor_list.append(tuple((i-1, j+1)))
                if (I[i, j] == I[i-1, j-1] and ((tuple((i-1, j-1)) in connection_list) == False)\
                and ((tuple((i-1, j-1)) in neighbor_list) == False)):
                    neighbor_list.append(tuple((i-1, j-1)))

            # This tuple has been verified, therefore its added to the
            # connection_list list and removed from the neighborhood list
            connection_list.append(neighbor_list.pop(0)) 
    connection_list.sort(key=lambda tup: tup[0]) # Sorting accoring to X
    connection_list.sort(key=lambda tup: tup[1]) # Soring according to Y
    return sorted(connection_list)

if __name__ == '__main__':
    # User input
    filename = input().rstrip()
    x_seed = int(input())
    y_seed = int(input())
    while(True):
        c = int(input())
        if (c == 4 or c == 8):
            break
        else:
            print('Invalid C, please either 4 or 8')

    # Using imageio
    I = (imageio.imread(filename) > 127).astype(np.uint8)

    # Applying the Flood Fill algorithm
    connection_list = flood_fill(I, x_seed, y_seed, c)

    # Output
    for i in connection_list:
        print(f'({i[0]} {i[1]})', end=' ')