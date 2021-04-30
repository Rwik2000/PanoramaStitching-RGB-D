# Code by Rwik Rana rwik.rana@iitgn.ac.in
'''
This is code to stitch image staken from various perspectives to output a
panorama image. In this code, I have used concepts of inverse homography, 
warping, RANSAC and Laplacian Blending to solve the problem statement.

To use the code, 
1. Add your dataset in the Dataset Directory.
2. In line 153, add your datasets in the Datasets array.
3. The output of the same would be save in the Output directory.

Flow of the Code:
1. SIFT to detect features.
2. Matching features using KNN
3. Finding the the homography matrix i.e. the mapping from one image plane to another.
4. warping the images using the inverse of homography matrix to 
   ensure no distortion/blank spaces remain in the transformed image.
5. Stitching and blending all the warped images together
'''
import numpy as np
import cv2
import os
import shutil
import imutils
import random

from numpy.lib.type_check import imag

from homography import homographyRansac
from Warp import Warp
from blend_and_stitch import stitcher

class panaroma_stitching():    
    def __init__(self):
        # Class parameters
        self.ismid = 1
        self.LoweRatio = 0.75
        self.warp_images = []
        self.warp_mask = []
        self.inbuilt_CV = 0
        self.p = 0
        self.dataset = "data"
        self.blendON = 1 #to use laplacian blend.... keep it ON!
    # Finding the common features between the two images, mapping the features, 
    # getting the homography and warping the required images
    def map2imgs(self, images):

        (imageB, imageA) = images
        # Finding features and their corresponding keypoints in the given images
        (kpsA, featuresA) = self.findSIFTfeatures(imageA)
        (kpsB, featuresB) = self.findSIFTfeatures(imageB)
        # match features between the two images
        H, invH= self.mapKeyPts(kpsA, kpsB,featuresA, featuresB)
        warpClass = Warp()
        # Warping the image
        warpClass.xOffset = 150
        warpClass.yOffset = 200
        mainimg = np.zeros((640,1000,3))
        mainimg[warpClass.xOffset:warpClass.xOffset+imageB.shape[0],warpClass.yOffset:warpClass.yOffset+imageB.shape[1]] = imageB
        warpedImg = warpClass.InvWarpPerspective(imageA, invH,H,(640, 1000))
        return np.uint8(warpedImg), np.uint8(mainimg)
    
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
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _, _image = cv2.threshold(_image, 1, 255, cv2.THRESH_BINARY)
        return _image

    def findSIFTfeatures(self, image):

        desc = cv2.SIFT_create()
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
            # Calling the homography class
            reprojThresh =4
            # Obtaining the homography matrix
            if not self.inbuilt_CV:
                homographyFunc = homographyRansac(reprojThresh,1000)
                H,_= homographyFunc.getHomography(ptsA, ptsB)
            else:
                H,_ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            
            # Obtaining the inverse homography matrix
            invH = np.linalg.inv(H)
            return H, invH
        return None
    
    def stitch2imgs(self, cimages):
        wimg,mimg  = self.map2imgs(cimages)
        warpedimgs = [mimg, wimg]
        warpedmasks = [self.extractMask(mimg), self.extractMask(wimg)]
        stitchDscp = stitcher()
        # Laplacian Blending is NOT USED. normal stitching is done.
        img = stitchDscp.stitchOnly(warpedimgs,warpedmasks)
        return img
        
        

def main(dataset, ref_number):     
    type = 'im'
    cimgs = [cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_0.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_1.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_2.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_3.jpg')]

    for i in range(len(cimgs)):
        cimgs[i] = imutils.resize(cimgs[i], width=300)

    inpImgs = cimgs[ref_number:ref_number+2]
    panoStitch = panaroma_stitching()
    result = panoStitch.stitch2imgs(inpImgs)
    print("========>Done! Final Image Saved in Outputs Dir!\n\n")
    cv2.imwrite(dataset+'_'+str(ref_number)+'_'+str(ref_number+1)+'.jpg', result)


dnames = ['0029','0292','0705','1524','2812','3345']
dname = '0292'
ref_image = 2
main(dname, 0)
# for name in dnames:
#     for k in range(3):
#         print('**********' + name + '**********' )
#         try:
#             print(k,k+1)
#             main(name, k)
#             print('done!!!!!!! ##################')
#         except:
#             pass


 
