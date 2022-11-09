// ComputerVision.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <fstream>

#include <numeric>
#include <vector>
#include <algorithm>
#include "cv.h"

#define NUMOF_FRAMES 900
#define CROP_PERCENT 0.2
#define THRESHOLD 15
#define MIN_NUMOF_WPIXEL 30
#define MIN_IRISCONTOURAREA 100

using namespace cv;
using namespace std;
using namespace ml;
using namespace cv::face;

CV::CV() {}

Mat CV::rotate(Mat src, double angle) //rotate function returning mat object with parametres imagefile and angle    
{
    Mat dst; //Mat object for output image file
    Point2f pt(src.cols / 2., src.rows / 2.); //point from where to rotate    
    Mat r = getRotationMatrix2D(pt, angle, 1.0); //Mat object for storing after rotation
    warpAffine(src, dst, r, Size(src.cols, src.rows)); ///applie an affine transforation to image.
    return dst; //returning Mat object for output image file
}

Rect CV::getLeftmostEye(vector<Rect>& eyes)
{
    int leftmost = 99999999;
    int leftmostIndex = -1;
    for (int i = 0; i < eyes.size(); i++) {
        if (eyes[i].tl().x < leftmost) {
            leftmost = eyes[i].tl().x;
            leftmostIndex = i;
        }
    }
    return eyes[leftmostIndex];
}

Rect CV::detectEyes(Mat& frame, CascadeClassifier& eyeCascade)
{
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
    equalizeHist(gray, gray); // enchance image contrast
    // Detect Both Eyes
    vector<Rect> eyes;
    eyeCascade.detectMultiScale(gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(90, 90)); // eye size (Size(90,90)) is determined emperically based on eye distance
    if (eyes.size() != 2) { // if both eyes not detected
        cout << "Error: Both eyes not detected" << endl;
        return Rect(0, 0, 0, 0); // return empty rectangle
    }
    for (Rect& eye : eyes) {
        rectangle(frame, eye.tl(), eye.br(), Scalar(0, 255, 0), 2); // draw rectangle around both eyes
    }
    // printf("eyex1 = %d, eyey1 = %d, eyew1 = %d, eyeh1 = %d\n", eyes[0].x, eyes[0].y, eyes[0].size().width, eyes[0].size().height);
    // printf("eyex2 = %d, eyey2 = %d, eyew2 = %d, eyeh2 = %d\n", eyes[1].x, eyes[1].y, eyes[1].size().width, eyes[1].size().height);

    imshow("frame", frame);
    return getLeftmostEye(eyes);
}

void CV::detectBlink(Mat& frame, Rect& eye, Rect& iris)
{
    static float prev_numof_bwpixels;
    static float curr_numof_bwpixels;
    static int eyestate;

    // BGR to Binary
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
    equalizeHist(gray, gray); // enchance image contrast
    Mat blur;
    GaussianBlur(gray, blur, Size(9, 9), 0); // blur image
    Mat thresh;
    threshold(blur, thresh, THRESHOLD, 255, THRESH_BINARY_INV); // convert to binary image

    // Crop Sides to Remove Eyebrows etc.
    int x = thresh.cols * CROP_PERCENT;
    int y = thresh.rows * CROP_PERCENT;
    int src_w = thresh.cols * (1 - (CROP_PERCENT * 2));
    int src_h = thresh.rows * (1 - (CROP_PERCENT * 2));
    Mat crop = thresh(Rect(x, y, src_w, src_h)); // crop side to remove eyebrows etc.

    // Get Iris
    int iris_h = iris.br().y - iris.tl().y;

    // Get Upper Half of Cropped Frame
    int upper_w = crop.cols;
    int upper_h = iris.br().y - (thresh.rows * CROP_PERCENT) - (iris_h * 0.2);
    Mat upper = crop(Rect(0, 0, upper_w, upper_h)); // get upper half of image

    // Calculate Histogram
    int histSize = 256;
    float range[] = { 0, 256 }; // the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat hist;
    calcHist(&upper, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate); // get histogram

    // Compare Current and Previous Frames
    prev_numof_bwpixels = curr_numof_bwpixels;
    curr_numof_bwpixels = hist.at<float>(255) < MIN_NUMOF_WPIXEL ? 0 : hist.at<float>(255);
    float percentDiff = ((prev_numof_bwpixels - curr_numof_bwpixels) / ((prev_numof_bwpixels + curr_numof_bwpixels) / 2)) * 100;
    if (percentDiff >= 100.0) {
        eyestate = 1;
        // printf("[%d] Close\n", frameno);
    }
    else if (percentDiff <= -20.0) {
        eyestate = 0;
        // printf("[%d] Open\n", frameno);
    }

    // Calculate PERCLOS
    calculate(hist, eyestate);

    // Draw Lines
    Point p1(x, y);
    Point p2(x + src_w, y + src_h);
    rectangle(gray, p1, p2, Scalar(0, 255, 0), 2);
    line(gray, Point(x, y + upper_h), Point(x + src_w, y + upper_h), Scalar(0, 0, 255), 2, 8, 0);

    imshow("gray", gray);
    imshow("crop", crop);
    imshow("upper", upper);
    waitKey(1);
}

// Iris Detection Steps:
//   1. Use detected eye, crop unwanted areas (i.e. eyebrows) by cropping the sides by x%
//   2. Determine largest contour which is the pupil
//   3. Limitation: Iris color should be on the black side 
Rect CV::detectIris(Mat& frame, Rect& eye)
{
    frame = frame(eye);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // Find contours
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
    equalizeHist(gray, gray); // enchance image contrast
    Mat blur;
    GaussianBlur(gray, blur, Size(9, 9), 0); // blur image
    Mat thresh;
    threshold(blur, thresh, THRESHOLD, 255, THRESH_BINARY_INV); // convert to binary image
    findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    if (contours.size() == 0) {
        cout << "Eyeball not detected" << endl;
        return Rect(0, 0, 0, 0);
    }

    int maxarea_contour_id = getMaxAreaContourId(contours);
    vector<Point> it = contours[maxarea_contour_id];
    contours.clear();
    contours.push_back(it);

    if (contourArea(contours.at(0)) < MIN_IRISCONTOURAREA)
    {
        cout << "Eyeball not detected" << endl;
        return Rect(0, 0, 0, 0);
    }

    // Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Point2f>center(contours.size());
    vector<float>radius(contours.size());

    for (int i = 0; i < contours.size(); i++)
    {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
        minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
    }

    // Draw polygonal contour + bonding rects + circles
    for (int i = 0; i < contours.size(); i++)
    {
        drawContours(frame, contours_poly, i, Scalar(95, 191, 0), 2, 8, vector<Vec4i>(), 0, Point());
        rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 191), 2, 8, 0);
        // circle(frame, center[i], (int)radius[i], color, 2, 8, 0);
    }

    // Show in a window
    imshow("thresh", thresh);
    return boundRect[0];
}

void CV::calculate(Mat& hist, int& eyestate)
{
    if (eyestates.size() == NUMOF_FRAMES)
    {
        int closedstates = 0;
        closedstates = accumulate(eyestates.begin(), eyestates.end(), 0);
        PERCLOS = (float)closedstates / NUMOF_FRAMES * 100;
        // printf("Closed states = %d PERCLOS = %f\n", closedstates, PERCLOS);

        std::rotate(eyestates.begin(), eyestates.begin() + 1, eyestates.end());
        eyestates.pop_back();
    }
    if (eyestate)
    {
        eyestates.push_back(1);
    }
    else
    {
        eyestates.push_back(0);
    }
}

int CV::getMaxAreaContourId(vector <vector<cv::Point>> contours)
{
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++)
    {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea)
        {
            maxArea = newArea;
            maxAreaContourId = j;
        } // End if
    } // End for
    cout << maxArea << endl;
    return maxAreaContourId;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
