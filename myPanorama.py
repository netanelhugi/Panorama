import cv2 as cv
import numpy as np
import imageRegistration as ir
import panorama_stitching as ps
import matplotlib.pyplot as plt


if __name__ == '__main__':


    img1 = cv.imread('./data/inp/mine/ariel1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('./data/inp/mine/ariel2.jpg', cv.IMREAD_GRAYSCALE)
    img3 = cv.imread('./data/inp/mine/ariel3.jpg', cv.IMREAD_GRAYSCALE)
    img4 = cv.imread('./data/inp/mine/ariel4.jpg', cv.IMREAD_GRAYSCALE)
    ariel = [img1,img2,img3,img4]

    images = ariel

    hpair = []

    # For every pair of images:
    for im in range(len(images)-1):
        # 2. Find and match features
        mp = ir.findMatchFeaturs(images[im], images[im+1])
        pts1, pts2 = ir.spiltPointsList(mp)

        # 3. Register homography pairs (using your ransacHomography)
        h, pts = ir.ransacHomography(pts1, pts2, numIters=10, inlierTol=5)

        # 4. Display inlier and outlier matches (using your displayMatches):
        # ir.displayMatchs(images[im], images[im+1], pts1, pts2, pts)

        hpair.append(h)

    # 5. Transform the homographies (using your accumulateHomographies):
    acc = ps.accumulateHomographies(hpair, 1)

    # 6. Load the RGB frames:
    img1 = cv.imread('./data/inp/mine/ariel1.jpg')
    img2 = cv.imread('./data/inp/mine/ariel2.jpg')
    img3 = cv.imread('./data/inp/mine/ariel3.jpg')
    img4 = cv.imread('./data/inp/mine/ariel4.jpg')
    ariel = [img1, img2, img3, img4]

    images = ariel
    # images = backyard


    # 7. Render panorama of each color channel (using your renderPanorama):
    Rimages = []
    Gimages = []
    Bimages = []

    # For every image:
    # a. convert from BGR to RGB.
    # b. Divide into 3 channels: [R],[G],[B]
    for i in range(len(images)):
        img = images[i]
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        r = img[::, ::, 0]
        g = img[::, ::, 1]
        b = img[::, ::, 2]

        Rimages.append(r)
        Gimages.append(g)
        Bimages.append(b)

    # Make panorama for each channel:
    Rpano = ps.renderPanorama(Rimages,acc)
    Gpano = ps.renderPanorama(Gimages,acc)
    Bpano = ps.renderPanorama(Bimages,acc)

    # print(Rpano)
    new_image = np.zeros((Rpano.shape[0], Rpano.shape[1], 3), dtype=int)
    new_image[::,::,0] = Rpano
    new_image[::,::,1] = Gpano
    new_image[::,::,2] = Bpano

    # Stack RGB together:
    rgbPano = np.dstack([Rpano,Gpano,Bpano])

    # # Display panorama image:
    # plt.imshow(new_image)
    # plt.show()

    # Convert to BGR:
    new_image = np.zeros((Rpano.shape[0], Rpano.shape[1], 3), dtype=int)
    new_image[::, ::, 0] = Bpano
    new_image[::, ::, 1] = Gpano
    new_image[::, ::, 2] = Rpano
    new_image.astype(np.uint8)

    cv.imwrite("./data/out/mine/arielPano.jpg", new_image)


    # arielBack images:

    img1 = cv.imread('./data/inp/mine/arielBack1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('./data/inp/mine/arielBack2.jpg', cv.IMREAD_GRAYSCALE)
    img3 = cv.imread('./data/inp/mine/arielBack3.jpg', cv.IMREAD_GRAYSCALE)
    img4 = cv.imread('./data/inp/mine/arielBack4.jpg', cv.IMREAD_GRAYSCALE)
    img5 = cv.imread('./data/inp/mine/arielBack4.jpg', cv.IMREAD_GRAYSCALE)
    ariel = [img1, img2, img3, img4,img5]

    images = ariel

    hpair = []

    # For every pair of images:
    for im in range(len(images) - 1):
        # 2. Find and match features
        mp = ir.findMatchFeaturs(images[im], images[im + 1])
        pts1, pts2 = ir.spiltPointsList(mp)

        # 3. Register homography pairs (using your ransacHomography)
        h, pts = ir.ransacHomography(pts1, pts2, numIters=10, inlierTol=5)

        # 4. Display inlier and outlier matches (using your displayMatches):
        # ir.displayMatchs(images[im], images[im+1], pts1, pts2, pts)

        hpair.append(h)

    # 5. Transform the homographies (using your accumulateHomographies):
    acc = ps.accumulateHomographies(hpair, 1)

    # 6. Load the RGB frames:
    img1 = cv.imread('./data/inp/mine/arielBack1.jpg')
    img2 = cv.imread('./data/inp/mine/arielBack2.jpg')
    img3 = cv.imread('./data/inp/mine/arielBack3.jpg')
    img4 = cv.imread('./data/inp/mine/arielBack4.jpg')
    img5 = cv.imread('./data/inp/mine/arielBack5.jpg')
    ariel = [img1, img2, img3, img4,img5]

    images = ariel
    # images = backyard

    # 7. Render panorama of each color channel (using your renderPanorama):
    Rimages = []
    Gimages = []
    Bimages = []

    # For every image:
    # a. convert from BGR to RGB.
    # b. Divide into 3 channels: [R],[G],[B]
    for i in range(len(images)):
        img = images[i]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        r = img[::, ::, 0]
        g = img[::, ::, 1]
        b = img[::, ::, 2]

        Rimages.append(r)
        Gimages.append(g)
        Bimages.append(b)

    # Make panorama for each channel:
    Rpano = ps.renderPanorama(Rimages, acc)
    Gpano = ps.renderPanorama(Gimages, acc)
    Bpano = ps.renderPanorama(Bimages, acc)

    # print(Rpano)
    new_image = np.zeros((Rpano.shape[0], Rpano.shape[1], 3), dtype=int)
    new_image[::, ::, 0] = Rpano
    new_image[::, ::, 1] = Gpano
    new_image[::, ::, 2] = Bpano

    # Stack RGB together:
    rgbPano = np.dstack([Rpano, Gpano, Bpano])

    # # Display panorama image:
    # plt.imshow(new_image)
    # plt.show()

    # Convert to BGR:
    new_image = np.zeros((Rpano.shape[0], Rpano.shape[1], 3), dtype=int)
    new_image[::, ::, 0] = Bpano
    new_image[::, ::, 1] = Gpano
    new_image[::, ::, 2] = Rpano
    new_image.astype(np.uint8)

    cv.imwrite("./data/out/mine/arielBackPano.jpg", new_image)









