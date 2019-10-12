import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imageRegistration as ir


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

    n = len(Hpair)  # Image number
    Htot = []

    # Build #n homography matrices between coordinate systems of img #mat and #m:
    for mat in range(n+1):

        # If #mat < #m:
        #     H[i,m] = H[m-1,m] * ... * H[mat+1,mat+2] * H[mat,mat+1]
        if mat < m:
            i = m - 1
            H_im = Hpair[i]

            while i > mat:
                i -= 1
                H_im = np.matmul(H_im, Hpair[i])

            # Noramilize the matrix and round to 5 numbers after the dot:
            H_im /= H_im[2,2]
            H_im = np.asarray([[round(j,5) for j in i] for i in H_im])
            Htot.append(H_im)

        # If #mat > #m:
        #     H[i,m] = H[m,m+1]^(-1) * ... * H[mat-2,mat-1]^(-1) * H[mat-1,mat]^(-1)
        if mat > m:
            i = m
            H_im = np.linalg.inv(Hpair[i])

            while i < mat - 1:
                i += 1
                invH = np.linalg.inv(Hpair[i])
                H_im = np.matmul(H_im, invH)

            # Noramilize the matrix and round to 5 numbers after the dot:
            H_im /= H_im[2,2]
            H_im = np.asarray([[round(j,5) for j in i] for i in H_im])
            Htot.append(H_im)

        # If #mat = #m:
        #   H[i,m] = [[1,0,0],[0,1,0],[0,0,1]
        if mat == m:
            H_im = np.identity(3)
            Htot.append(H_im)

    return Htot


'''
function name: renderPanorama
% RENDERPANORAMA Renders a set of images into a combined panorama image.
% Arguments:
% im − array or dict of n grayscale images.
% H − array or dict array of n 3x3 homography matrices transforming the ith image
% coordinates to the panorama image coordinates.
% Returns:
% panorama − A grayscale panorama image composed of n vertical strips that
% were backwarped each from the relevant frame im{i} using homography H{i}.
'''
def renderPanorama(im, H):

    numOfImages = len(im)
    Hcorners = []
    # Hcenters = []
    edges = []

    # If we have just one image, return:
    if numOfImages == 1:
        panorama = im[0]

        return panorama

    # For every image from 'im':
    for i in range(numOfImages):

        rowsSize, colsSize = im[i].shape
        curHomography = H[i]

        # Find the corners of the image after the transformation:
        corners = np.array([[0, 0], [colsSize, 0], [0, rowsSize], [colsSize, rowsSize]])
        Hcorners.append(ir.applyHomography(corners, curHomography))

        # find new edges of image
        HcornersNp = np.asarray(Hcorners[i])

        curr_edges = []
        curr_edges.append(min(HcornersNp[::, 0]))  # minX
        curr_edges.append(max(HcornersNp[::, 0]))  # maxX
        curr_edges.append(min(HcornersNp[::, 1]))  # minY
        curr_edges.append(max(HcornersNp[::, 1]))  # maxY
        edges.append(curr_edges)

    edgesNp = np.asarray(edges)

    # X range
    panoramaColSize = int(max(edgesNp[::,1])-min(edgesNp[::, 0])+1)
    # Y range
    panoramaRowSize = int(max(edgesNp[::,3])-min(edgesNp[::, 2])+1)

    panorama = np.zeros((panoramaRowSize, panoramaColSize))

    for i in range(numOfImages):

        Hmat = H[i]
        # warp images and add to 'panorama':
        panorama += cv.warpPerspective(im[i], Hmat,panorama.shape).T

        # Crop 'panorama':
        ans = panorama.T
        yBorder = int(max(edgesNp[::,3]))
        xBorder = int(max(edgesNp[::,1]))
        ans = ans[:yBorder,:xBorder]

    return ans


def generatePanorama(imagesNames):

    # 1. Read grayscale frames
    imagesGray = []
    for name in imagesNames:
        imagesGray.append(cv.imread(name,cv.IMREAD_GRAYSCALE))

    hpair = []

    # For every pair of images:
    for im in range(len(imagesGray)-1):
        # 2. Find and match features
        mp = ir.findMatchFeaturs(imagesGray[im], imagesGray[im+1])
        pts1, pts2 = ir.spiltPointsList(mp)

        # 3. Register homography pairs (using your ransacHomography)
        h, pts = ir.ransacHomography(pts1, pts2, numIters=500, inlierTol=10)

        # 4. Display inlier and outlier matches (using your displayMatches):
        ir.displayMatchs(imagesGray[im], imagesGray[im+1], pts1, pts2, pts)

        hpair.append(h)

    # 5. Transform the homographies (using your accumulateHomographies):
    acc = accumulateHomographies(hpair, 1)

    # 6. Load the RGB frames:
    imagesRGB = []
    for name in imagesNames:
        imagesRGB.append(cv.imread(name))

    # 7. Render panorama of each color channel (using your renderPanorama):
    Rimages = []
    Gimages = []
    Bimages = []

    # For every image:
    # a. convert from BGR to RGB.
    # b. Divide into 3 channels: [R],[G],[B]
    for i in range(len(imagesRGB)):
        img = imagesRGB[i]
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        r = img[::, ::, 0]
        g = img[::, ::, 1]
        b = img[::, ::, 2]

        Rimages.append(r)
        Gimages.append(g)
        Bimages.append(b)

    # Make panorama for each channel:
    Rpano = renderPanorama(Rimages,acc)
    Gpano = renderPanorama(Gimages,acc)
    Bpano = renderPanorama(Bimages,acc)

    # Stack RGB together:
    rgbPano = np.stack([Rpano,Gpano,Bpano],axis=-1).astype(int)

    # Display panorama image:
    plt.imshow(rgbPano)
    plt.show()


if __name__ == '__main__':

    oxfordImagesNames = []
    oxfordImagesNames.append("./data/inp/examples/oxford1.jpg")
    oxfordImagesNames.append("./data/inp/examples/oxford2.jpg")

    officeImagesNames = []
    officeImagesNames.append('./data/inp/examples/office1.jpg')
    officeImagesNames.append('./data/inp/examples/office2.jpg')
    officeImagesNames.append('./data/inp/examples/office3.jpg')
    officeImagesNames.append('./data/inp/examples/office4.jpg')

    backyardImagesNames = []
    backyardImagesNames.append('./data/inp/examples/backyard1.jpg')
    backyardImagesNames.append('./data/inp/examples/backyard2.jpg')
    backyardImagesNames.append('./data/inp/examples/backyard3.jpg')

    generatePanorama(officeImagesNames)




