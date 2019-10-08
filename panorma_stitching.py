import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def accumulateHomographies(Hpair,m):
    """
    Accumulate homography matrix sequence.
    :param1 Hpair 􀀀 array or dict of M􀀀1 3x3 homography matrices where Hpairfig is a homography that transforms between coordinate systems i and i+1.
    :param2 m -Index of coordinate system we would like to accumulate the given homographies towards (see details below).
    """
    H=[]
    if len(Hpair)==2: #2 frames
        H1 =np.matmul(Hpair[1],Hpair[1])
        H2 =np.identity(3)
        H3 =np.matmul(np.linalg.inv(Hpair[0]),np.linalg.inv(Hpair[1]))
        H.append(H1)
        H.append(H2)
        H.append(H3)
        return H
        # CASE  i < m
        for i in range(m+1):
            H_im = Hpair[m]
            k = m - 1
            while k >= i:
                H_im = np.matmul(H_im, Hpair[k])
                k -= 1
            # Htemp.append(H_im)
            H.append(H_im)

        # CASE 2  i = m
           if i == m:
              H_im = np.identity(3)
              H.append(H_im)
              i += 1

        # CASE 3 i > m
        for i in range(m + 1, len(Hpair) - 1):
            k = m + 1
            H_im = np.linalg.inv(Hpair[m])
            while k <= i:
                inverseH = np.linalg.inv(Hpair[k])
                H_im = np.matmul(H_im, inverseH)
                k += 1
            H.append(H_im)

        return H





