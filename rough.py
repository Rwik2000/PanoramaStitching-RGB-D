import numpy as np
import cv2
import os
import shutil
import imutils
import random
# from numpy.lib.type_check import imag

from homography import homographyRansac
from Warp import Warp
from blend_and_stitch import stitcher
from discretize import discretize


class panaroma_stitching():    
    def __init__(self):
        # Class parameters
        self.ismid = 1
        self.LoweRatio = 0.8
        self.warp_images = []
        self.warp_mask = []
        self.inbuilt_CV = 0
        self.p = 0
        self.dataset = "data"
        self.blendON = 0 #to use laplacian blend.... keep it ON!
    # Finding the common features between the two images, mapping the features, 
    # getting the homography and warping the required images
    def map2imgs(self, images):

        (imageB, imageA) = images
        # Finding features and their corresponding keypoints in the given images
        (kpsA, featuresA) = self.findSIFTfeatures(imageA)
        (kpsB, featuresB) = self.findSIFTfeatures(imageB)

        # match features between the two images
        H, invH= self.mapKeyPts(kpsA, kpsB,featuresA, featuresB)
        # print(H,invH)
        warpClass = Warp()
        if self.ismid:
            # Setting the offset only for the main image... homography takes care of the rest of the images
            warpClass.xOffset = 150
            warpClass.yOffset = 200
        # Warping the image
        warpedImg = warpClass.InvWarpPerspective(imageA, invH,H,(640, 1000))
        cv2.imshow('x',np.uint8(warpedImg))
        cv2.waitKey(0)
        return np.uint8(warpedImg)


    def extractMask(self,image):

        _image = image.copy()
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _, _image = cv2.threshold(_image, 1, 255, cv2.THRESH_BINARY)
        return _image

    def findSIFTfeatures(self, image):

        desc = cv2.ORB_create()
        (kps, features) = desc.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def mapKeyPts(self, kpsA, kpsB, featuresA, featuresB):
        desc = cv2.DescriptorMatcher_create("BruteForce")
        # using KNN to find the matches.
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
                homographyFunc = homographyRansac(reprojThresh,1000)
                H= homographyFunc.getHomography(ptsA, ptsB)
            else:
                H,_ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            
            # print(H)
            # Obtaining the inverse homography matrix
            invH = np.linalg.inv(H)
            # print(invH)
            return H, invH
        return None
    
    def MultiStitch(self, images):
        
        if len(images)%2!=0:
            mid_image_loc = len(images)//2 
        else:
            mid_image_loc = len(images)//2 - 1
        
        left_images = images[:mid_image_loc]
        left_images = left_images[::-1]

        right_images = images[mid_image_loc+1:]
        mid_image = self.map2imgs((images[mid_image_loc], images[mid_image_loc]))
        self.ismid = 0
        temp_mid_image = mid_image
        self.p = 0
        for i in range(len(left_images)):
            print("=============> Transformed Image : ", self.p)
            self.p+=1
            temp_warp = self.map2imgs((temp_mid_image, left_images[i]))
            _mask = self.extractMask(temp_warp)
            self.warp_mask.append(_mask)
            self.warp_images.append(temp_warp)
            temp_mid_image = temp_warp
        self.warp_images = self.warp_images[::-1]
        self.warp_mask = self.warp_mask[::-1]
        print("=============> Transformed Image : ", self.p)
        self.p+=1
        self.warp_images.append(mid_image)
        _mask = self.extractMask(mid_image)
        self.warp_mask.append(_mask)
        temp_mid_image = mid_image
        for i in range(len(right_images)):
            print("=============> Transformed Image : ", self.p)
            self.p+=1
            temp_warp = self.map2imgs((temp_mid_image, right_images[i]))
            _mask = self.extractMask(temp_warp)
            self.warp_mask.append(_mask)
            self.warp_images.append(temp_warp)
            temp_mid_image = temp_warp
        print("Transformations Applied....")
        print("Blending and Stitching....")

        stitchDscp = stitcher()
        if self.blendON:
            final = stitchDscp.LaplacianBlend(self.warp_images,self.warp_mask, n=5)
        else:
            final = stitchDscp.stitchOnly(self.warp_images,self.warp_mask)
        cv2.imshow("stitch",final)        
        cv2.waitKey(0)
        return final
        

disc = discretize(10)
dname = '2812'
type = 'depth'
img_p0 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_0.jpg')
img_p1 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_1.jpg')
img_p2 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_2.jpg')
# img_p3 = cv2.imread('../RGBD dataset/00000'+dname+'/'+type+'_3.jpg')

images=[img_p0,img_p1,img_p2]

images = disc.exec_multi(images)

for i in range(len(images)):
    images[i] = imutils.resize(images[i], width=300)

inpImgs = images[:]
panoStitch = panaroma_stitching()
result = panoStitch.MultiStitch(inpImgs)


