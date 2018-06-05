#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace cv;
using namespace std;

RNG rng(12345);

struct contour_sorter // 'less' for contours
{
    bool operator ()( const vector<Point>& a, const vector<Point> & b )
    {
        Rect ra(boundingRect(a));
        Rect rb(boundingRect(b));
        // scale factor for y should be larger than img.width
        return ( (ra.x + 2000*ra.y) < (rb.x + 2000*rb.y) );
    }
};

int main()
{
	// Load source image
	string filename = "images/himanshu.png";
	Mat src = imread(filename);

	// Check if image is loaded fine
	if (!src.data) {
		cerr << "Problem loading image!" << endl;
	}

	// Resizing for practical reasons
	Mat rsz;
	double rszScale;
	if (max(src.size().height, src.size().width) > 1500) {
		rszScale = 1500 / double(max(src.size().height, src.size().width));
	} else {
		rszScale = 1;
	}
	resize(src, rsz, Size(), rszScale, rszScale);

	// Transform source image to gray if it is not
	Mat gray;
	if (rsz.channels() == 3) {
		cvtColor(rsz, gray, CV_BGR2GRAY);
	} else {
		gray = rsz;
	}

	// Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
	Mat bw;
	adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);

	// Create the images that will use to extract the horizonta and vertical lines
	Mat horizontal = bw.clone();
	Mat vertical = bw.clone();

	int scale = 20; // play with this variable in order to increase/decrease the amount of lines to be detected

	// Specify size on horizontal axis
	int horizontalsize = horizontal.cols / scale;

	// Create structure element for extracting horizontal lines through morphology operations
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize,1));

	// Apply morphology operations
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	// dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1)); // expand horizontal lines

	// Specify size on vertical axis
	int verticalsize = vertical.rows / scale;

	// Create structure element for extracting vertical lines through morphology operations
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1,verticalsize));

	// Apply morphology operations
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));

	// create a mask which includes the tables
	Mat mask = horizontal + vertical;

	// find the joints between the lines of the tables, we will use this information
	// in order to descriminate tables from pictures (tables will contain more than 4 joints
	// while a picture only 4 (i.e. at the corners))
    Mat joints;
	bitwise_and(horizontal, vertical, joints);

    // Find external contours from the mask, which most probably will belong to tables or to images
    vector<Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
	cv::findContours(mask, contours, hierarchy, RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// for( size_t i = 0; i< contours.size(); i++ )
    // {
    //     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    //     drawContours( rsz, contours, i, color, 2, 8, hierarchy, 0, Point() );
    // }
	// imshow("contours", rsz);
	// waitKey();

    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Mat> rois;

    for (size_t i = 0; i < contours.size(); i++)
    {
		if (hierarchy[i][2] == -1)
			continue;

        // find the area of each contour
        double area = contourArea(contours[i]);

		// filter individual lines of blobs that might exist and they do not represent a table
        if(area < 100) // value is randomly chosen, you will need to find that by yourself with trial and error procedure
            continue;

        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));

        // find the number of joints that each table has
		Mat roi = joints(boundRect[i]);

        vector<vector<Point> > joints_contours;
        findContours(roi, joints_contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

        // if the number is not more than 5 then most likely it not a table
        if(joints_contours.size() <= 4)
			continue;

		rois.push_back(rsz(boundRect[i]).clone());

		// drawContours(rsz, contours, i, Scalar(0, 0, 255), CV_FILLED, 8, vector<Vec4i>(), 0, Point());
		// rectangle(rsz, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 1, 8, 0);
		break;
    }

	Mat timetableImage = rois[0];
	Mat ttGray, ttBw, ttCopy;
	ttCopy = timetableImage.clone();
	cvtColor(timetableImage, ttGray, CV_BGR2GRAY);
	adaptiveThreshold(~ttGray, ttBw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	hierarchy.clear();
	contours.clear();
	cv::findContours(ttBw, contours, hierarchy, RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	contours_poly.clear();
	contours_poly.resize(contours.size());
	boundRect.clear();
	boundRect.resize(contours.size());
	rois.clear();

	std::sort(contours.begin(), contours.end(), contour_sorter());

	for (size_t i = 0; i < contours.size(); i++) {
		if (hierarchy[i][2] == -1) {
			double area = contourArea(contours[i]);
			if (area < 250)
				continue;
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        	boundRect[i] = boundingRect(Mat(contours_poly[i]));
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        	// drawContours( ttCopy, contours, i, color, 2, 8, hierarchy, 0, Point() );
		rectangle(ttCopy, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0);
		rois.push_back(timetableImage(boundRect[i]).clone());
		}
	}

	tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();
	// Initialize tesseract-ocr with English, without specifying tessdata path
	if (tess->Init(NULL, "eng")) {
	     fprintf(stderr, "Could not initialize tesseract.\n");
	     exit(1);
	}
	for (size_t i=0; i<rois.size(); ++i) {
		imshow("yolo", rois[i]);
		waitKey();
		tess->SetImage((uchar*)rois[i].data, rois[i].size().width, rois[i].size().height, rois[i].channels(), rois[i].step1());
		tess->Recognize(0);
		const char* out = tess->GetUTF8Text();
		cout << out << endl;
	}
	tess->End();
	return 0;
}
