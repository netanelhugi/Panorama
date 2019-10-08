import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


'''
% function name: findMatchFeaturs
% Finding match features betweem 2 images, using OpenCV methods.
% Arguments:
% img1,img2 − images.
% Returns:
% match_points - List of matching points between the 2 images.
'''
def findMatchFeaturs(img1,img2):

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ratio = 0.75

    # Apply ratio test
    goodImg1 = []
    goodImg2 = []

    match_points = []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            goodImg1.append([m])
            goodImg2.append([n])
            p1 = kp1[m.queryIdx].pt
            p1s = [p1[0],p1[1]]
            p2 = kp2[m.trainIdx].pt
            p2s = [p2[0],p2[1]]
            match_points.append([p1s,p2s])

    return match_points


'''
% function name: applyHomography
% APPLYHOMOGRAPHY Transform coordinates pos1 to pos2 using homography H.
% Arguments:
% pos1 - An nx2 matrix of [x,y] point coordinates per row.
% H - A 3x3 homography matrix.
% Returns:
% pos2 - An nx2 matrix of [x,y] point coordinates per row obtained from transforming pos1 using H.
'''
def applyHomography(pos1, H):

    pos2 = []  # For the new points positions after homography transformation

    # For every point calculate transformation:
    for p in pos1:

        p = [p[0],p[1],1]  # Convert to homogeneous coordinate
        newP = H.dot(p)  # Multiply in the transformation matrix
        newP = [newP[0]/newP[2],newP[1]/newP[2]]  # Convert from homogeneous coordinate

        pos2.append(newP)

    return pos2


'''
% function name: computeHomography
% Calculate homography between 4 matching positions(8 points), using DLT.
% Arguments:
% pts1 - An 4x2 matrix of [x,y] point coordinates per row, positions from the first image.
% pts2 - An 4x2 matrix of [x,y] point coordinates per row, matching positions on the other image.
% Returns:
% h - A 3x3 homography matrix.
'''
def computeHomography(pts1,pts2):

    mat = []

    for p1,p2 in zip(pts1,pts2):
        # DLT equations:
        arr1 = [p1[0],p1[1],1,0,0,0,-p2[0]*p1[0],-p2[0]*p1[1],-p2[0]]
        arr2 = [0,0,0,p1[0],p1[1],1,-p2[1]*p1[0],-p2[1]*p2[1],-p2[1]]

        mat.append(arr1)
        mat.append(arr2)

    # convert A to array:
    A = np.asarray(mat)
    # Compute the SVD:
    U, S, Vh = np.linalg.svd(A)
    # The parameters are in the last line of Vh and normalize them:
    L = Vh[-1, :] / Vh[-1, -1]
    # homography matrix:
    h = L.reshape(3, 3)

    return h

'''
% function name: computeDist
% Calculate squared euclidean distance between 2 points(the real point to estimate point).
% Arguments:
% pos1 - An nx2 matrix of [x,y] point coordinates per row, positions from the first image.
% pos2 - An nx2 matrix of [x,y] point coordinates per row, positions from the second image.
% H - A 3x3 homography matrix.
% inlierTol − inlier tolerance threshold.
% Returns:
% inlierPts - List oh inlier points.
'''
def computeDist(pos1,pos2,H,inlierTol):

    inlierPts = []  # For points whose distance is less than the threshold(inlier points).

    # Estimated position of the points from img1 in img2 according to the homography matrix:
    pts1est = applyHomography(pos1,H)

    # For every tuple of (estimated point,real point(img1),real point(img2),
    # calculate squared euclidean distance, if the norm is less than the threshold,
    # enter the real points in the inlierPts list.
    for est,p1,p2 in zip(pts1est,pos1,pos2):

        pEst = np.transpose([est[0], est[1]])
        pReal = np.transpose([p2[0], p2[1]])

        E = pReal - pEst
        norm = np.linalg.norm(E)

        if norm < inlierTol:
            inlierPts.append([p1,p2])

    return inlierPts


'''
function name: ransacHomography
% RANSACHOMOGRAPHY Fit homography to maximal inliers given point matches using the RANSAC algorithm.
% Arguments:
% pos1,pos2 − Two nx2 matrices containing n rows of [x,y] coordinates of matched points.
% numIters − Number of RANSAC iterations to perform.
% inlierTol − inlier tolerance threshold.
% Returns:
% bestH − A 3x3 normalized homography matrix.
% inliersPts − An array containing the indices in pos1/pos2 of the maximal set of inlier matches found.
'''
def ransacHomography(pos1,pos2,numIters,inlierTol):

    # For the best result during the iterations:
    bestH = []  # For homography matrix.
    inliersPts = []  # For the inlier points.

    for iter in range(numIters):
        # For each iteration, select 4 random points:
        idx = np.random.choice(len(pos1), size=4, replace=False)
        batch1 = []
        batch2 = []
        for n in idx:
            batch1.append(pos1[n])
            batch2.append(pos2[n])

        # Compute homography matrix:
        h = computeHomography(batch1,batch2)
        # Find inlier points:
        inlierPtsCurr = computeDist(pos1,pos2,h,inlierTol)

        # If the number of inlier points in the current iteration is greater than what we have saved,
        # save the matrix and the list of inlier points.
        if len(inlierPtsCurr) > len(inliersPts):
            inliersPts = inlierPtsCurr
            bestH = h

    return bestH,inliersPts

'''
function name: spiltPointsList
% Split a list of points from the following form: [[p1img1,p1img2],[p2img1,p2img2],...]
% for two lists, one for each image
% Arguments:
% pts - lists of points. [[p1img1,p1img2],[p2img1,p2img2],...]
% Returns:
% img1Pts- list of points. [p1img1,p2img1,...]
img2Pts - list of points.  [p1img2,p2img2,...]
'''
def spiltPointsList(pts):
    img1Pts = []
    img2Pts = []
    for i in pts:
        img1Pts.append(i[0])
        img2Pts.append(i[1])

    return img1Pts,img2Pts


'''
function name:displayMatches
% DISPLAYMATCHES Display matched pt. pairs overlayed on given image pair.
% Arguments:
% im1,im2 − two grayscale images
% pos1,pos2 − nx2 matrices containing n rows of [x,y] coordinates of matched
% points in im1 and im2 (i.e. the i’th match’s coordinate is
% pos1(i,:) in im1 and and pos2(i,:) in im2).
% inlind − k−element array of inlier matches (e.g. see output of
% ransacHomography)
'''
def displayMatchs(im1,im2,pos1,pos2,inlind):

    h,w = img1.shape

    # Remove duplicates between all points to inliers points:
    pos1and2 = zip(pos1,pos2)
    pos1and2 = [x for x in pos1and2 if x not in inlind]
    pos1,pos2 = spiltPointsList(pos1and2)

    vis = np.concatenate((im1, im2), axis=1)  # Merge the 2 images horizontally.

    # Arrange all points in lists of x and y for drawing in the image:
    x_list = [x for [x, y] in pos1]
    y_list = [y for [x, y] in pos1]
    x_list2 = [x+w for [x, y] in pos2]
    y_list2 = [y for [x, y] in pos2]
    x_list += x_list2
    y_list += y_list2

    plt.plot(x_list, y_list, 'ro',markersize=1)  # Draw each points in red.

    # Draw a blue line between every 2 matching outlier points:
    for p1,p2 in zip(pos1,pos2):
        plt.plot([p1[0],p2[0]+w],[p1[1],p2[1]],'b-',linewidth=0.7)

    # Draw a yellow line between every 2 matching inlier points:
    inline_img1, inline_img2 = spiltPointsList(inlind)
    for p1,p2 in zip(inline_img1,inline_img2):
        plt.plot([p1[0],p2[0]+w],[p1[1],p2[1]],'y-',linewidth=0.7)

    # Display images, points, and lines:
    plt.imshow(vis,cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    img1 = cv.imread('oxford1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('oxford2.jpg', cv.IMREAD_GRAYSCALE)

    mp = findMatchFeaturs(img1, img2)
    pts1, pts2 = spiltPointsList(mp)
    h, pts = ransacHomography(pts1, pts2, numIters=1000, inlierTol=10)
    displayMatchs(img1, img2, pts1, pts2, pts)

