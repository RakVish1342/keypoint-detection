/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        /* Ring Buffer Implementation */
        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        // Needs to only hold two images since a constant velocity model is being used. 
        // Else might need upto 3 or more (if acceleration etc model are used).
        dataBuffer.push_back(frame);
        if (dataBuffer.size() > dataBufferSize)
        {
            dataBuffer.erase(dataBuffer.end());
        }
        // cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT IMAGE KEYPOINTS */
        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = "SHITOMASI";
        //  string detectorType = "HARRIS";
        // string detectorType = "FAST";
        // string detectorType = "BRISK";
        // string detectorType = "ORB";
        // string detectorType = "AKAZE";
        // string detectorType = "SIFT";

        double keyTime = 0;
        if (detectorType.compare("SHITOMASI") == 0)
        {
            keyTime = detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            keyTime = detKeypointsHarris(keypoints, imgGray, false);
        }
        else if ( (detectorType.compare("FAST") == 0) ||
                (detectorType.compare("BRISK") == 0) ||
                (detectorType.compare("ORB") == 0) ||
                (detectorType.compare("AKAZE") == 0) ||
                (detectorType.compare("SIFT") == 0) )
        {
            keyTime = detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        else
        {
            std::cout << "NOT SUPORTED" << std::endl;
        }


        /* Maintaining keypoints of vehicle only */
        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        vector<cv::KeyPoint> vehicleKeypoints;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            for (cv::KeyPoint kp : keypoints)
            {
                if(vehicleRect.contains(kp.pt))
                {
                    vehicleKeypoints.push_back(kp);
                }
            }
            keypoints = vehicleKeypoints;
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        // cout << "#2 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */
        cv::Mat descriptors;
        string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        //string descriptorType = "BRIEF"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        // Check NOTE 1
        //string descriptorType = "ORB"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        // string descriptorType = "FREAK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        // Check NOTE 2
        // string descriptorType = "AKAZE"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        // Check NOTE 3
        // string descriptorType = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        keyTime += descKeypoints((dataBuffer.end() - 1)->keypoints, 
                                (dataBuffer.end() - 1)->cameraImg, 
                                descriptors, descriptorType);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;
        // cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */
            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            //string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            // string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
            // string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);


            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
            // cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();

                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                std::cout << "----------" << std::endl;
                std::cout << "Detector: " << detectorType << std::endl;
                std::cout << "Descriptor: " << descriptorType << std::endl;
                std::cout << "Total Keypoints: " << keypoints.size() << std::endl;
                std::cout << "Matched Keypoints: " << matches.size() << std::endl;
                std::cout << "Detection + Description Time (ms): " << keyTime*1000 << std::endl;
                std::cout << "(Matching time not calculated/included)" << std::endl;
                std::cout << "==========" << std::endl;

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}




// Debug Notes:

// NOTE 1:
// When SIFT detector and ORB Descriptor are used. OUT OF MEMORY runtime error appears!!
// Is fine if both ORB detector and descriptor are used.
/*
    root@42d94b09e2b4:/home/workspace/keypoint-detection/_build# ./2D_feature_tracking 
    #1 : LOAD IMAGE INTO BUFFER done
    #2 : DETECT KEYPOINTS done
    terminate called after throwing an instance of 'cv::Exception'
    what():  OpenCV(4.1.0) /opencv/modules/core/src/alloc.cpp:55: error: (-4:Insufficient memory) Failed to allocate 70166064384 bytes in function 'OutOfMemoryError'

    Aborted (core dumped)
*/

// NOTE 2:
// When using ORB and ORB its fine

        // Simlarly, when SIFT/ORB detectors are used with AKAZE descriptor, error appear
        // Works fine when both are AKAZE 
        /*
            root@42d94b09e2b4:/home/workspace/keypoint-detection/_build# ./2D_feature_tracking 
            #1 : LOAD IMAGE INTO BUFFER done
            #2 : DETECT KEYPOINTS done
            terminate called after throwing an instance of 'cv::Exception'
            what():  OpenCV(4.1.0) /opencv/modules/features2d/src/kaze/AKAZEFeatures.cpp:1192: error: (-215:Assertion failed) 0 <= kpts[i].class_id && kpts[i].class_id < static_cast<int>(evolution_.size()) in function 'Compute_Descriptors'

            Aborted (core dumped)
        */

//NOTE 3:
// SIFT detector with SIFT descriptor currently throwing this error:
/*
    root@42d94b09e2b4:/home/workspace/keypoint-detection/_build# ./2D_feature_tracking 
    #1 : LOAD IMAGE INTO BUFFER done
    #2 : DETECT KEYPOINTS done
    SIFT descriptor extraction in 107.006 ms
    #3 : EXTRACT DESCRIPTORS done
    #1 : LOAD IMAGE INTO BUFFER done
    #2 : DETECT KEYPOINTS done
    SIFT descriptor extraction in 78.1693 ms
    #3 : EXTRACT DESCRIPTORS done
    terminate called after throwing an instance of 'cv::Exception'
    what():  OpenCV(4.1.0) /opencv/modules/core/src/batch_distance.cpp:282: error: (-215:Assertion failed) (type == CV_8U && dtype == CV_32S) || dtype == CV_32F in function 'batchDistance'

    Aborted (core dumped)

    SOLVED: https://answers.opencv.org/question/10046/feature-2d-feature-matching-fails-with-assert-statcpp/
    Can't use Hamming/Binary descriptor matching technique for SIFT and SURF. 
    Have to set string descriptorType = "DES_HOG"; NOT string descriptorType = "DES_BINARY";
*/