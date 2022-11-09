#ifndef CV_H_
#define CV_H_

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
// #include "cv.h"

#define NUMOF_FRAMES 900
#define CROP_PERCENT 0.2
#define THRESHOLD 15
#define MIN_NUMOF_WPIXEL 30
#define MIN_IRISCONTOURAREA 100

using namespace cv;
using namespace std;
using namespace ml;
using namespace cv::face;

class CV
{
public:
	float PERCLOS = 0.0;

	CV();
	Mat rotate(Mat src, double angle);
	Rect getLeftmostEye(vector<Rect>& eyes);
	Rect detectEyes(Mat& frame, CascadeClassifier& eyeCascade);
	void detectBlink(Mat& frame, Rect& eye, Rect& iris);
	Rect detectIris(Mat& frame, Rect& eye);
	int getMaxAreaContourId(vector <vector<cv::Point>> contours);
	void calculate(Mat& hist, int& eyestate);

private:
	vector<float> whitepixels;
	vector<int> eyestates;
};

#endif