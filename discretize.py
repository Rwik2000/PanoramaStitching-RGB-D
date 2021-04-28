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
    
    def exec_multi(self,imgs):
        '''
        Returns a discretized images given one input images.

        --------------------
        Input :
            imgs -> list of grayscale images
        
        Output :
            disc_imgs ->list of discretized images according to the bins property provided in the class property 'bin'
        '''
        
        a = 0
        for i in range(len(imgs)):
            try:
                imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)

            except:
                pass
            a = max(a, np.amax(imgs[i]))
        step_size = a//self.bins        
        discrete_steps = [i*step_size for i in range(self.bins)]
        print(discrete_steps)
        disc_imgs = []
        for img in imgs:
            disc_img = img//step_size
            disc_img = disc_img*step_size
            disc_img = cv2.cvtColor(disc_img, cv2.COLOR_GRAY2BGR)
            disc_imgs.append(disc_img)
            cv2.imshow('x',disc_img)
            cv2.waitKey(0)

        return disc_imgs