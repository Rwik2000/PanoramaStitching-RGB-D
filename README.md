![alt text](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/Dataset/sample.jpg)
# Panorama Stitching

Code by Rwik Rana, IITGn.

View the project on https://github.com/Rwik2000/Panorama-Stitching-v2.0 for better viewing 

This is a Code to stitch multiple images to create panoramas.This is the tree for the entire repo:

```
Panorama Stitching RGB-D
        |
        |___Dataset
        |       |--I1
        |       |--I2 ....
        |
        |___results
        |       |-- hard-code
        |       |-- inbuilt
        |
        |___blend_and_stitch.py
        |___homography.py
        |___discretize.py
        |___rgb_panorama.py
        |___rgbd_panorama.py
        |___Warp.py

```

THis is a continuation of the repo: https://github.com/Rwik2000/Panorama-Stitching-v2.0 . The new additions to the code being using depth images to warp the images. In this repo, only 2 images are warped and stitched unlike the previous repo.

### To use the code, 

##### Libraries needed
* opencv
* numpy
* imutils
* tqdm

##### Procedure:
Use [rgbd_panorama.py](https://github.com/Rwik2000/PanoramaStitching-RGB-D/blob/main/rgbd_panorama.py) to get results using the depth image. Usage:

1. Add your dataset in the Dataset Directory.
2. In [line 184](https://github.com/Rwik2000/PanoramaStitching-RGB-D/blob/main/rgbd_panorama.py#L184), add your dataset name and the image you want to start with i.e. an input 2 will result in using the images 2 and 3 from the dataset. **Note** : please the proper naming scheme of the dataset and the corresponding depth and coloured images. The dataset used has a maximum of 3 images, so 2 is the maximum number to be put. The number of quantization/discrete levels are to be input in the `main()`. Default value is kept as 14.

```python
dname = '0705'
ref_image = 2

```
3. The output of the same would be saved 
4. In the terminal run `python rgbd_panorama.py`

### Flow of the Code:
1. Discretize and break the reference image according to input quantization level.
2. For each level of quantization, use  SIFT to detect features. The detection will between the quantized layer and the whole of source image.
3. Matching features using KNN
4. Finding the the homography matrix i.e. the mapping from one image plane to another.
5. warping each level using the corresponding homography matrices. images using the inverse of homography matrix to 
   ensure no distortion/blank spaces remain in the transformed image.
6. Stitching all the warped images together

#### Implementation of Ransac:
Refer to the `homographyRansac()` in [Homography.py](https://github.com/Rwik2000/PanoramaStitching-RGB-D/blob/main/homography.py). Access the object `getHomography()` of the class to get the desired homography matrix.
For my implementation, for each pair of image, RANSAC does 100 iterations and threshold is kept as 4.
```python
        # Set the threshold and number of iterations neede in RANSAC.
        self.ransacThresh = ransacThresh
        self.ransaciter = ransacIter


```
#### Warping
For Warping, use the `InvWarpPerspective()` from [Warp.py](https://github.com/Rwik2000/PanoramaStitching-RGB-D/blob/main/Warp.py).

**Note** : Avoid using the `warpPersepctive()`, because it gives distorted and incomplete projection of the images.

```python
def InvWarpPerspective(self, im, invA, H,output_shape):

        x1 = [0,0,1]
        x2 = [im.shape[1], im.shape[0],1]
        x1_trnsf = H.dot(np.array(x1).T)
        x1_trnsf = list(x1_trnsf/x1_trnsf[2])
        x2_trnsf = H.dot(np.array(x2).T)
        x2_trnsf = list(x2_trnsf/x2_trnsf[2])

        .
        .
        .
        .       
                
        return warpImage
```
#### Laplacian Blending
Input list of images and their corresponding masks in grayscale. Along with that add the number of layers of laplcian pyramid is to be made for the final blending and stitching. it is to be noted that the input images to the `LaplacianBlending()` function must have a shape such that the number of rows and columns are divisible by 2^n.

The shape of the output image can be changed .Notice here, I have kept a size of (640, 1000) i.e. height and width being 640 and 1000 respectively.

## Results

![alt text](https://github.com/Rwik2000/PanoramaStitching-RGB-D/blob/main/results/hard-code/0292/0292_0_1.jpg)

![alt text](https://github.com/Rwik2000/PanoramaStitching-RGB-D/blob/main/results/hard-code/2812/2812_1_2.jpg)

![alt text](https://github.com/Rwik2000/PanoramaStitching-RGB-D/blob/main/results/hard-code/3345/3345_1_2.jpg)


## Files:
* `rgb_panormama.py` to run and compile the entire code for only rgb warping.
* `rgbd_panormama.py` to run and compile the entire code for RGB-D warping.
* `Warp.py` contains `Warp()` clas which helps in image warping
* `homography.py` helps in finding the homography for two images. It also contains the code for RANSAC.
* `blend_and_stitch.py` contains `stitcher()` Class which helps in blending and stitching. You can either use `laplcacianBlending()` for blending and stitching or `onlyStitch()` to only stitch without blending.
* `dicretize.py` contains class `discretize()` and use `exec_multi()` to discretize multiple input images.

