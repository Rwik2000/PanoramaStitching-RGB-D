import numpy as np
import cv2
from tqdm import tqdm
from homography import homographyRansac
from Warp import Warp
from blend_and_stitch import stitcher
from discretize import discretize


class panaroma_stitching():    
    def __init__(self):
        # Class parameters
        self.LoweRatio = 0.75
        self.inbuilt_CV = 0 #Use inbuilt opencv functions instead of the hand-made functions
        self.isFast = 0
        self.blendON = 0 #to use laplacian blend.... keep it ON!

    def map2imgs(self, images):
        '''
        Finding the common features between the two images and the corresponding 
        keypoints.
        --------------------------
        INPUT -> images list [img1,img2]
        img1 is kept intact and img2 is transformed to stitch with img1
        --------------------------
        OUTPUT -> keypoints, features

        '''

        # image A is transformed
        (imageB, imageA) = images
        (kpsA, featuresA) = self.findSIFTfeatures(imageA)
        (kpsB, featuresB) = self.findSIFTfeatures(imageB)

        return kpsA,kpsB,featuresA,featuresB

    def extractMask(self,image):
        '''
        Extracting the ROI of the image within the overall region.
        '''
        _image = image.copy()
        _image = _image.astype('uint8')
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _, _image = cv2.threshold(_image, 1, 255, cv2.THRESH_BINARY)
        return _image

    def findSIFTfeatures(self, image):
        '''
        Finding features using SIFT.
        '''
        desc = cv2.SIFT_create(nfeatures=5000)
        (kps, features) = desc.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def mapKeyPts(self, kpsA, kpsB, featuresA, featuresB):
        '''Option of Flann is turned off by default to ensure accuracy is given more preference 
        than time. Turn on isFast to use Flann instead of Brute Force to reduce the time for processing'''
        if self.isFast:
            FLANN_INDEX_KDTREE = 3
            index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            _Matches = flann.knnMatch(featuresA, featuresB, 2)
        else:
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
            # Calling the homography class
            reprojThresh =4
            # Obtaining the homography matrix
            if not self.inbuilt_CV:
                homographyFunc = homographyRansac(reprojThresh,5000)
                H, _= homographyFunc.getHomography(ptsA, ptsB)

                invH = np.linalg.inv(H)
                return H,invH, len(matches)
            else:
                H,_ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh, maxIters=5000)
                invH = np.linalg.inv(H)
                return H,invH, len(matches)
        # Returning 0 matches if matches<4.
        return 0,0,0  

    def stitch2imgs(self,cimgs,disc_imgs,masks, dlvl):
        '''
        Given two images, their discretized/Quantized images and their corresponding masks.
        This function outputs a stitched image. The stitching is done in such a way that each 
        quantized level of the reference image is mapped to the source image. For each level,
        a homography matrix is obtained and thus each level is warped individually. Finally these 
        warped parts are stitched together to give the final output.

        Input -> coloured images, discretised images, masks, number of levels

        Output -> Stitched image
        '''
        kpsA,kpsB,featuresA,featuresB = self.map2imgs(cimgs)
        warpClass = Warp()
        # Warping the main image into a black canvas.
        mimg = np.zeros((640,1600,3))
        warpClass.xOffset = 150
        warpClass.yOffset = 200
        mimg[150:150+cimgs[0].shape[0],200:200+cimgs[0].shape[1]] = cimgs[0]

        # making lists of warped images and their corresponding masks
        warpedimgs = [mimg]
        warpedmasks = [self.extractMask(mimg)]

        # Recording the preceding homography and inverse-homography matrices.
        prevH = None
        previnvH = None
        # Iterating over all the depth levels.
        print('Homography and Warping each level')
        for i in tqdm(range(dlvl), total = dlvl):
            pts = []
            feat = []
            # Iterating over all the keypoints and checking which keypoint lies in the masked 
            # region for the reference image. Adding the corresponding keypoints and features 
            # to the pts and feat lists for further matching and homography calculations.
            for j in range(len(kpsA)):
                pt = np.uint8(kpsA[j])
                if masks[1][i][pt[0]][pt[1]] == 255:

                    pts.append(list(kpsA[j]))
                    feat.append(featuresA[j])

            H, invH, matches = self.mapKeyPts(np.array(pts), kpsB, np.array(feat), featuresB)
            # Using previous homography and inverse-homogrpahy matrices if the total matches is less than 10.
            # The value of 10 obtained after some tweaks and iterations.
            if matches<10:
                H = prevH
                invH = previnvH

            # Getting the warped image and masks and addign them to the matrices for further stitching
            temp_warpedImg = warpClass.InvWarpPerspective(disc_imgs[1][i],invH,H,(640, 1600))
            warpedimgs.append(np.uint8(temp_warpedImg))
            warpedmasks.append(self.extractMask(temp_warpedImg))
            # Updating previous homograhy matrices
            prevH = H
            previnvH = invH
            # print("Level "+str(i)+" Warped ....")
        print('Start Stitching.....')
        stitchDscp = stitcher()
        # Laplacian Blending is NOT USED. normal stitching is done.
        img = stitchDscp.stitchOnly(warpedimgs,warpedmasks)
        print('stitching done!')
        return img

def main(dataset, ref_number,levels = 14):     
    type = 'depth'
    dimgs = [cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_0.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_1.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_2.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_3.jpg')]

    type = 'im'
    cimgs = [cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_0.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_1.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_2.jpg'),
             cv2.imread('../RGBD dataset/00000'+dataset+'/'+type+'_3.jpg')]

    dimages = dimgs[ref_number:ref_number+2]
    cimages = cimgs[ref_number:ref_number+2] 

    disc = discretize(levels)
    disc_imgs, masks = disc.exec_multi(dimages,cimages)
    print("Discretized/Quantized Images ....")

    pano = panaroma_stitching()
    final = pano.stitch2imgs(cimages, disc_imgs,masks,disc.bins)
    cv2.imwrite(dataset+'_'+str(ref_number)+'_'+str(ref_number+1)+'.jpg', final)

# dnames = ['0029','0292','0705','1524','2812','3345']
dname = '0705'
ref_image = 2
main(dname,ref_image)





