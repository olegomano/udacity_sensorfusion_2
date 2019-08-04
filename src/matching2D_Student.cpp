#include <numeric>
#include "matching2D.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {   //DES_BINARY, DES_HOG
        if(descriptorType == "DES_BINARY"){
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        }else if(descriptorType == "DES_HOG"){
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);      
        }
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descriptorType == "DES_HOG"){
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        }else if(descriptorType == "DES_BINARY"){
            matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(5, 24, 2)).create();
            if(descSource.type()!=CV_32F || descRef.type()!=CV_32F) {
               descSource.convertTo(descSource, CV_32F);
               descRef.convertTo(descRef, CV_32F);  
            }
        }
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        std::vector<std::vector<cv::DMatch>> knnMatches; //https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        matcher->knnMatch(descSource,descRef,knnMatches,2);

        for(int i = 0; i < knnMatches.size(); i++){
            if(knnMatches[i][0].distance < 0.8f * knnMatches[i][1].distance ){
                matches.push_back(knnMatches[i][0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// BRIEF, ORB, FREAK, AKAZE, SIFT
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }else if(descriptorType == "BRIEF"){
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }else if(descriptorType == "ORB"){
        extractor = cv::ORB::create();
    }else if(descriptorType == "FREAK"){
        extractor = cv::xfeatures2d::FREAK::create();
    }else if(descriptorType == "SIFT"){
        extractor = cv::xfeatures2d::SIFT::create();
    }else if(descriptorType == "AKAZE"){
        extractor = cv::AKAZE::create();
    }


    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    cv::Mat keypointMatrix = cv::Mat::zeros(img.size(),CV_32FC1);
    cv::cornerHarris(img,keypointMatrix,3,7,0.05);
    cv::imshow("harris",keypointMatrix);

    for(int x = 0; x < keypointMatrix.rows; x++){
        for(int y = 0; y < keypointMatrix.cols; y++){
            float value = keypointMatrix.ptr<float>()[x*keypointMatrix.cols + y];
            if(value > 0.001){
                keypoints.push_back(cv::KeyPoint(cv::Point2f(y,x),4));
            }
        }
    }

    if (bVis){
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);

    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
    //FAST, BRISK, ORB, AKAZE, SIFT
    if(detectorType == "FAST"){
        cv::FAST(img,keypoints,20);
    }else if(detectorType == "BRISK"){
        auto detector = cv::BRISK::create();
        detector.get()->detect(img,keypoints);      
    }else if(detectorType == "ORB"){
        auto detector = cv::ORB::create();
        detector.get()->detect(img,keypoints);      
    }else if(detectorType == "AKAZE"){
        auto detector = cv::AKAZE::create();
        detector.get()->detect(img,keypoints);      
    }else if(detectorType == "SIFT"){
        auto detector = cv::xfeatures2d::SIFT::create();
        detector.get()->detect(img,keypoints);      
    }

    if (bVis){
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Modern";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);

    }
}



void filterKeypoints(std::vector<cv::KeyPoint>& keypoints, const cv::Rect2f& rect){
    for(int i = 0; i < keypoints.size(); i++){
        if(!rect.contains(keypoints[i].pt)){
            cv::KeyPoint tmp = keypoints[i];
            keypoints[i]     = keypoints[keypoints.size() - 1];
            keypoints.erase(keypoints.end());
            --i;
        }
    }
}