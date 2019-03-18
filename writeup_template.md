## Writeup for Advanced Lane Finding Project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_chessboard.png "Undistorted"
[image2]: ./output_images/test_image_undistorted.png "Undistorted test image"
[image3]: ./examples/Histogram_helper.png "Histogram_helper"
[image4]: ./examples/S_channel_sbinary_x.png "color channel thresholding"
[image5]: ./examples/color_grad.png "Color and Gradient thresholded"
[image6]: ./examples/Grad_X_Y_mag_dir.png "Grad_X_Y_mag_dir"
[image7]: ./examples/perspective_transformed.png "Perspective transform"
[image8]: ./examples/histogram.png "Histogram"
[image9]: ./examples/Warped_binary_image.png "Output"
[image10]: ./examples/using_sliding_window.png "Output"
[image11]: ./examples/Using_Search_around_poly.png "Output"
[image12]: ./examples/Overlay_lanes.png "Output"


[video1]: ./output_with_sliding.mp4 "Video Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.    

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in `./project.ipynb` in `undistort_img()` and `undistort()` function where we calibrate in `undistort_img()` and store results in `./camera_cal.p` pickle file and use it undistort images in `undistort()` function.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted-Chessboard-Image][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Undistorted test image][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in `color_grad()`  in 6th cell of IPyhton notebook . Various methods and threshold values were tested in finding put best alternative and learn behaviour of each channel kept in seprate IPython notebook .

```python
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
h_channel = hls[:,:,0]
l_channel = hls[:,:,1]
s_channel = hls[:,:,2]
```
Using this we can seperate each channel data.
```python 
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
```
We are then combining output from two or more thresholded matrix:
```python 
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
```
.  Here are few example of my output for various alternatives we got.

![Grad_X_Y_mag_dir thresolding][image6]
![color channel thresholding][image4]
![Color and Gradient thresholded][image5]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

    The code for my perspective transform includes a function called `perspective_transform()`, which appears in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`) and returns a transformed image `binary_warped`.
We have used source (`src`) and destination (`dst`) points to transform the perspective of image .  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(203,720),(1099,720),(707,463),(580,463)])
dst = np.float32([(203,720),(1099,720),(1099,0),(203,0)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 203,720       | 203,720       | 
| 1099,720      | 1099,720      |
| 707,463       | 1099,0        |
| 580,463       | 203, 0        |

We use `cv2.getPerspectiveTransform(src, dst)`
`cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)`  functions to achieve 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective transform][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
You can find code for finding lane line pixels in function `find_lane_pixels()` in cell 8 in same IPython notebook .
First , I take histogram of bottom part of image `binary_warped[binary_warped.shape[0]//2:,:]` 
```python
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
```
![Perspective_binary][image3]

![Histogram][image8]

By taking max values from both sides we get a head start and we search using we can use the two highest peaks from our histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go .
In `find_lane_pixels()` we take these base points, we'll want to loop for `nwindows`, with the given window sliding left or right if it finds the mean position of activated pixels within the window to have shifted.
This looks something like this in action :
![using_sliding_window][image10]

Once we have some intution about lines we can now take values from previous frame computations and use for current frame for finding lanes
code can be reffered by `search_around_poly()` funtion which takes in image and left and right lane fits for fit we have `fit_polynomial()`
in same cell .
Results for this function looks like this :
![Using_Search_around_poly][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `get_curve()` function in 11th cell . We have used  ym_per_pix = 30.5/720 # meters per pixel in y dimension
xm_per_pix = 3.7/720 # meters per pixel in x dimension as conversions to measure lane distance in meters . Finally we calculate the radius of curve in (m) meter . For vehicle position we calculate its distance from center of lane and function returns all three values .   

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 8th cell in  in the function named `draw_lanes()`. We reversed back the image to normal perspective and drawing lines on top of it using points we found in lane finding step . Here is an example of my result on a test image:

![Overlay_lanes][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_with_sliding.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

It took too much time in playing around with gradient thresholding and color thresholding and finding the right thresholds but still we can find good arguments in changing it for better output . Learnt new techniques for processing images . First to calibrate the camera so that we could udistort the images . Camera lens produce distortions while capturing images and thresholding lane lines which were mostly yellow and white which takes down to seprate these color from other . There were shadows of trees and sunlight in some images for which i needed to combine multiple thresholding techniques as gradient alone could not get any lane lines . Pipeline may likely fail for too curved roads which have sharp turns in edge of images . To make this more robust we could use deep learning techniques for better results 
