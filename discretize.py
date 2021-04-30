import cv2
import numpy as np

class discretize():

    def __init__(self, bins):
        """
        This class discretizes the GRAY SCALE (1 channel image) into bins

        ----------
        Class properties ---> bins (number of disrete levels)

        """
        self.bins = bins
    
    def exec_single(self,img):
        '''
        Returns a single discretized image given one input image.

        --------------------
        Input :
            img -> grayscale image (1 channel image)
        
        Output :
            disc -> discretized image according to the bins property provided in the class property 'bin'
        '''
        a = np.amax(img)
        step_size = a//self.bins
        disc_img = img//step_size
        disc_img = disc_img*step_size
        disc_img = cv2.cvtColor(disc_img, cv2.COLOR_GRAY2BGR)

        return disc_img
    
    def exec_multi(self,dimgs, cimgs):
        '''
        Returns a discretized images given one input images.

        --------------------
        Input :
            imgs -> list of grayscale images
        
        Output :
            disc_imgs ->list of discretized images according to the bins property provided in the class property 'bin'
        '''
        
        a = 0
        for i in range(len(dimgs)):
            # Converting to 1 channel image
            dimgs[i] = cv2.cvtColor(dimgs[i], cv2.COLOR_BGR2GRAY)
            a = max(a, np.amax(dimgs[i]))
        step_size = a//self.bins        
        discrete_steps = [i*step_size for i in range(self.bins)]
        disc_imgs = []
        masks = []
        for i in range(len(cimgs)):
            depthimgs = []
            maskimgs = []
                
            disc_img = dimgs[i]//step_size
            disc_img = disc_img*step_size
            disc_img = cv2.cvtColor(disc_img, cv2.COLOR_GRAY2BGR)
            # dicretizing and maskibg the coloured image
            for j in range(0,len(discrete_steps)):
                p = discrete_steps[j]
                tempMask = cv2.inRange(disc_img, np.array([p]*3), np.array([p]*3)) 
                masked = cv2.bitwise_and(cimgs[i], cimgs[i], mask=tempMask)
                depthimgs.append(masked)
                maskimgs.append(tempMask)

            disc_imgs.append(depthimgs)
            masks.append(maskimgs)

        return disc_imgs, masks