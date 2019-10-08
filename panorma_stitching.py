import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

'''
function name: accumulateHomographies
% ACCUMULATEHOMOGRAPHY Accumulate homography matrix sequence.
% Arguments:
% Hpair − array or dict of M−1 3x3 homography matrices where Hpair{i} is a
% homography that transforms between coordinate systems i and i+1.
% m − Index of coordinate system we would like to accumulate the
% given homographies towards (see details below).
% Returns:
% Htot − array or dict of M 3x3 homography matrices where Htot{i} transforms
% coordinate system i to the coordinate system having the index m.
% Note:
% In this exercise homography matrices should always maintain
% the property that H(3,3)==1. This should be done by normalizing them as
% follows before using them to perform transformations H = H/H(3,3).
'''


def accumulateHomographies(Hpair, m):

    n = len(Hpair)
    Htot = []

    for mat in range(n+1):

        # CASE 1 -  i < m
        if mat < m:
            i = m - 1
            H_im = Hpair[i]

            while i > mat:
                i -= 1
                H_im = np.matmul(H_im, Hpair[i])

            Htot.append(H_im)

        # CASE 2 -  i > m
        if mat > m:
            i = m
            H_im = np.linalg.inv(Hpair[i])

            while i < mat - 1:
                i += 1
                # print(i)
                invH = np.linalg.inv(Hpair[i])
                H_im = np.matmul(H_im, invH)

            Htot.append(H_im)

        # CASE 3  i = m
        if mat == m:
            H_im = np.identity(3)
            Htot.append(H_im)

    return Htot


h1 = [[1, -1, -2], [2, -3, -5], [-1, 3, 5]]
h2 = [[1, -1, -2], [2, -3, -5], [-1, 3, 5]]
h3 = [[1, -1, -2], [2, -3, -5], [-1, 3, 5]]
h4 = [[1, -1, -2], [2, -3, -5], [-1, 3, 5]]
h5 = [[1, -1, -2], [2, -3, -5], [-1, 3, 5]]

# h = [h1, h2, h3, h4, h5]
h = [h1, h2]

acc = accumulateHomographies(h, math.ceil(len(h) / 2))
print(len(acc))
