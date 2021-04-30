import numpy as np
import cv2
import os
import shutil
import imutils
import random

from numpy.linalg.linalg import inv
# from numpy.lib.type_check import imag

from homography import homographyRansac
from Warp import Warp
from blend_and_stitch import stitcher
from discretize import discretize


class panaroma_stitching():    
    def __init__(self):
        # Class parameters
        self.ismid = 1
        self.LoweRatio = 0.75
        self.warp_images = []
        self.warp_mask = []
        self.inbuilt_CV = 0
        self.blendON = 0 #to use laplacian blend.... keep it ON!
    # Finding the common features between the two images, mapping the features, 
    # getting the homography and warping the required images
    def map2depths(self, images):

        (imageB, imageA) = images
        (kpsA, featuresA) = self.findSIFTfeatures(imageA)
        (kpsB, featuresB) = self.findSIFTfeatures(imageB)

        A = self.mapKeyPts(kpsA, kpsB,featuresA, featuresB)
        if A is not None:
            return A
        else:
            return 0,0,-100


    def extractMask(self,image):

        _image = image.copy()
        _image = _image.astype('uint8')
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _, _image = cv2.threshold(_image, 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('mask',_image)
        # cv2.waitKey(0)
        return _image

    def findSIFTfeatures(self, image):

        desc = cv2.SIFT_create()
        (kps, features) = desc.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def mapKeyPts(self, kpsA, kpsB, featuresA, featuresB):
        # FLANN_INDEX_KDTREE = 3
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
        # search_params = dict(checks=100)
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # _Matches = flann.knnMatch(featuresA, featuresB, 2)

        desc = cv2.DescriptorMatcher_create("BruteForce")
        _Matches = desc.knnMatch(featuresA, featuresB, 2)

        matches = []
        # loop over the raw matches
        for m in _Matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.LoweRatio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])      
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # print(ptsA, ptsB)
            # Calling the homography class
            reprojThresh =4
            # Obtaining the homography matrix
            if not self.inbuilt_CV:
                homographyFunc = homographyRansac(reprojThresh,5000)
                H, hScore= homographyFunc.getHomography(ptsA, ptsB)
                # print(H)
                # print('ik')
                invH = np.linalg.inv(H)
                return H,invH, hScore
            else:
                H,_ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh, maxIters=5000)
                invH = np.linalg.inv(H)
                return H,invH, 0
        return None    

    def stitch2imgs(self,cimages,disc_imgs, dlvl):
        w= []
        final_H = None
        final_invH = None
        check = 0
        maxScore = 0
        Hc,invHc,_ = self.map2depths(cimages)
        for i in reversed(range(dlvl)):
            cv2.imshow('d',disc_imgs[0][i])
            cv2.waitKey(0)
            H, invH, hScore = self.map2depths([cimages[0], disc_imgs[1][i]])
            if hScore>=maxScore or check==0:
                check=1
                maxScore = hScore
                final_H = H
                final_invH = invH
        warpClass = Warp()
        final_H = Hc
        final_invH = invHc
        
        # Setting the offset only for the main image... homography takes care of the rest of the images
        mimg = np.zeros((640,1000,3))
        warpClass.xOffset = 150
        warpClass.yOffset = 200
        mimg[150:150+cimages[0].shape[0],200:200+cimages[0].shape[1]] = cimages[0]
        # Warping the image
        warpedImg = warpClass.InvWarpPerspective(cimages[1], final_invH,final_H,(640, 1000))
        mask = [self.extractMask(mimg),self.extractMask(warpedImg)]
        stitchDscp = stitcher()
        final = stitchDscp.stitchOnly([mimg, warpedImg], mask)
        cv2.imshow('j', np.uint8(final))
        cv2.waitKey(0)

    def stitch2imgs2(self,cimages,disc_imgs, dlvl):
        w= []
        Hs = []
        invHs = []
        hScores = []
        final_H = None
        final_invH = None

        # maxScore = 0
        for i in range(dlvl):
            H, invH, hScore = self.map2depths([disc_imgs[0][i], disc_imgs[1][i]])
            Hs.append(H)
            invHs.append(invH)
            hScores.append(hScore)
        
        final_H, final_invH = self.finalHomography(Hs,invHs,hScores)

        warpClass = Warp()

        mimg = np.zeros((640,1000,3))
        warpClass.xOffset = 150
        warpClass.yOffset = 200
        mimg[150:150+cimages[0].shape[0],200:200+cimages[0].shape[1]] = cimages[0]
        # Warping the image
        warpedImg = warpClass.InvWarpPerspective(cimages[1], final_invH,final_H,(640, 1000))
        mask = [self.extractMask(mimg),self.extractMask(warpedImg)]
        stitchDscp = stitcher()
        final = stitchDscp.stitchOnly([mimg, warpedImg], mask)
        cv2.imshow('j', np.uint8(final))
        cv2.waitKey(0)
    
    def finalHomography(self,Hs,invHs,hScores):
        h = np.zeros_like(Hs[0])
        invh = np.zeros_like(invHs[0])
        for i in range(len(hScores)):
            h+=Hs[i]*hScores[i]**2
            invh+=invHs[i]*hScores[i]**2
        
        h = h/np.sum(np.square(hScores))
        invh = invh/np.sum(np.square(hScores))
        # print(h)
        return h, invh
        # print(h)
    def newStitch2imgs(self,cimgs,disc_imgs,dlvl):
        Hs = []
        invHs = []
        warpClass = Warp()
        mimg = np.zeros((640,1000,3))
        warpClass.xOffset = 150
        warpClass.yOffset = 200
        mimg[150:150+cimgs[0].shape[0],200:200+cimgs[0].shape[1]] = cimgs[0]

        warpedimgs = [mimg]
        warpedmasks = [self.extractMask(mimg)]
        for i in (range(dlvl)):
            # cv2.imshow('d',disc_imgs[1][i])
            # cv2.imshow('m',cimages[0])
            # cv2.waitKey(0)
            try:
                H, invH, _ = self.map2depths([cimgs[0], disc_imgs[1][i]])
                # print(H,invH)
                temp_warpedImg = warpClass.InvWarpPerspective(disc_imgs[1][i],invH,H,(640, 1000))
                cv2.imshow('d',np.uint8(temp_warpedImg))
                # cv2.imshow('m',cimages[0])
                cv2.waitKey(0)
                warpedimgs.append(np.uint8(temp_warpedImg))
                warpedmasks.append(self.extractMask(temp_warpedImg))
            except:
                pass
        
        stitchDscp = stitcher()
        img = stitchDscp.stitchOnly(warpedimgs,warpedmasks)
        cv2.imshow('m',np.uint8(img))
        cv2.waitKey(0)


disc = discretize(3)
dname = '0292'
type = 'depth'
dimg_p0 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_0.jpg')
dimg_p1 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_1.jpg')
dimg_p2 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_2.jpg')
dimg_p3 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_3.jpg')

type = 'im'
cimg_p0 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_0.jpg')
cimg_p1 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_1.jpg')
cimg_p2 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_2.jpg')
cimg_p3 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_3.jpg')

dimages = [dimg_p0,dimg_p1]
cimages = [cimg_p0,cimg_p1]

# cv2.imshow('d',dimages[0])
# cv2.waitKey(0)
disc_imgs = disc.exec_multi(dimages,cimages)

pano = panaroma_stitching()
final = pano.stitch2imgs(cimages, disc_imgs,disc.bins)




