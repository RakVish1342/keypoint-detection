# SFND 2D Feature Tracking

This project works to detect keypoints from a car that is driving in front, describe the keypoints, and match them to corresponding keypoints in the next camera frame. The following aspects of the implementation are discussed below:

i. Keypoint Detection <br/>
ii. Keypoint Description <br/>
iii. Keypoint Matching <br/>
iv. Performance Analysis <br/>


## i. Keypoint Detection
Found like this:
<img src="images/keypoints.png" width="820" height="248" />

Restricted to only car


## ii. Keypoint Description


## iii. Keypoint Matching


<img src="images/keypoints_mapping.png" width="820" height="248" />



## iv. Performance Analysis

time, matched keypoints








## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.
