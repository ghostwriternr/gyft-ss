#include <algorithm>
#include <iostream>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

using namespace cv;
using namespace std;

RNG rng(12345);
int tt_rows = 6;
int tt_cols = 10;

struct contour_sorter {
  bool operator()(const pair<Mat, vector<Point>> &a,
                  const pair<Mat, vector<Point>> &b) {
    Rect ra(boundingRect(a.second));
    Rect rb(boundingRect(b.second));
    if (fabs(ra.y - rb.y) < 5)
      return ra.x < rb.x;
    return ra.y < rb.y;
  }
};

struct y_sorter {
  bool operator()(const pair<Mat, vector<Point>> &a,
                  const pair<Mat, vector<Point>> &b) {
    Rect ra(boundingRect(a.second));
    Rect rb(boundingRect(b.second));
    return ra.y < rb.y;
  }
};

struct point_sorter {
  bool operator()(Point pt1, Point pt2) { return (pt1.y < pt2.y); }
};

vector<int> group_cells(vector<pair<Mat, vector<Point>>> roi_contours) {
  vector<int> starts = {0};
  vector<pair<Mat, vector<Point>>> inter = roi_contours;
  vector<pair<int, int>> inter_points;
  for (size_t i = 0; i < inter.size(); ++i) {
    inter_points.push_back(make_pair(boundingRect(inter[i].second).y, i));
  }
  for (size_t i = inter_points.size() - 1; i > 0; --i) {
    inter_points[i].first = inter_points[i].first - inter_points[i - 1].first;
  }
  sort(inter_points.begin(), inter_points.end());
  for (size_t i = inter_points.size() - tt_rows + 1; i < inter_points.size();
       ++i) {
    starts.push_back(inter_points[i].second);
  }
  sort(starts.begin(), starts.end());
  for (size_t i = 0; i < starts.size(); ++i) {
    cout << starts[i] << " ";
  }
  cout << endl;
  return starts;
}

int main(int argc, char **argv) {
  // Load source image
  if (argc != 2) {
    cerr << "Incorrect number of arguments!" << endl;
  }

  string filename(argv[1]);
  cout << filename << endl;
  Mat src = imread(filename);

  // Check if image is loaded fine
  if (!src.data) {
    cerr << "Problem loading image!" << endl;
  }

  // Resizing for practical reasons
  Mat scaled, rsz;
  double rszScale;
  rszScale = 1500 / double(max(src.size().height, src.size().width));
  resize(src, scaled, Size(), rszScale, rszScale);
  fastNlMeansDenoising(scaled, scaled);
  if (rszScale > 1) {
    GaussianBlur(scaled, rsz, cv::Size(0, 0), 3);
    addWeighted(scaled, 1.5, rsz, -0.5, 0, rsz);
  } else {
    rsz = scaled;
  }

  // Transform source image to gray if it is not
  Mat gray;
  if (rsz.channels() == 3) {
    cvtColor(rsz, gray, CV_BGR2GRAY);
  } else {
    gray = rsz;
  }

  // Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
  Mat bw;
  adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
                    15, -2);

  // Create the images that will use to extract the horizonta and vertical lines
  Mat horizontal = bw.clone();
  Mat vertical = bw.clone();

  int scale = 20; // play with this variable in order to increase/decrease the
                  // amount of lines to be detected

  // Specify size on horizontal axis
  int horizontalsize = horizontal.cols / scale;

  // Create structure element for extracting horizontal lines through morphology
  // operations
  Mat horizontalStructure =
      getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));

  // Apply morphology operations
  erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
  dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
  // dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1)); //
  // expand horizontal lines

  // Specify size on vertical axis
  int verticalsize = vertical.rows / scale;

  // Create structure element for extracting vertical lines through morphology
  // operations
  Mat verticalStructure =
      getStructuringElement(MORPH_RECT, Size(1, verticalsize));

  // Apply morphology operations
  erode(vertical, vertical, verticalStructure, Point(-1, -1));
  dilate(vertical, vertical, verticalStructure, Point(-1, -1));

  // create a mask which includes the tables
  Mat mask = horizontal + vertical;

  // find the joints between the lines of the tables, we will use this
  // information in order to descriminate tables from pictures (tables will
  // contain more than 4 joints while a picture only 4 (i.e. at the corners))
  Mat joints;
  bitwise_and(horizontal, vertical, joints);

  // Find external contours from the mask, which most probably will belong to
  // tables or to images
  vector<Vec4i> hierarchy;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, hierarchy, RETR_CCOMP,
                   CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  // for( size_t i = 0; i< contours.size(); i++ )
  // {
  //     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255),
  //     rng.uniform(0,255) ); drawContours( rsz, contours, i, color, 2, 8,
  //     hierarchy, 0, Point() );
  // }
  // imshow("contours", rsz);
  // waitKey();

  vector<vector<Point>> contours_poly(contours.size());
  vector<Rect> boundRect(contours.size());
  vector<Mat> rois;

  for (size_t i = 0; i < contours.size(); i++) {
    if (hierarchy[i][2] == -1)
      continue;

    // find the area of each contour
    double area = contourArea(contours[i]);

    // filter individual lines of blobs that might exist and they do not
    // represent a table
    if (area < 100) // value is randomly chosen, you will need to find that by
                    // yourself with trial and error procedure
      continue;

    approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
    boundRect[i] = boundingRect(Mat(contours_poly[i]));

    // find the number of joints that each table has
    Mat roi = joints(boundRect[i]);

    vector<vector<Point>> joints_contours;
    findContours(roi, joints_contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    // if the number is not more than 5 then most likely it not a table
    if (joints_contours.size() <= 4)
      continue;

    rois.push_back(rsz(boundRect[i]).clone());

    // drawContours(rsz, contours, i, Scalar(0, 0, 255), CV_FILLED, 8,
    // vector<Vec4i>(), 0, Point()); rectangle(rsz, boundRect[i].tl(),
    // boundRect[i].br(), Scalar(0, 255, 0), 1, 8, 0);
    break;
  }

  Mat timetableImage = rois[0];
  Mat ttGray, ttBw, ttCopy;
  ttCopy = timetableImage.clone();
  cvtColor(timetableImage, ttGray, CV_BGR2GRAY);
  adaptiveThreshold(~ttGray, ttBw, 255, CV_ADAPTIVE_THRESH_MEAN_C,
                    THRESH_BINARY, 15, -2);
  hierarchy.clear();
  contours.clear();
  cv::findContours(ttBw, contours, hierarchy, RETR_CCOMP,
                   CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
  contours_poly.clear();
  contours_poly.resize(contours.size());
  boundRect.clear();
  boundRect.resize(contours.size());
  rois.clear();

  vector<pair<Mat, vector<Point>>> roi_contours;

  for (size_t i = 0; i < contours.size(); i++) {
    if (hierarchy[i][2] == -1) {
      double area = contourArea(contours[i]);
      // cout << area << " " << hierarchy[i] << endl;
      if (area < 800 || area > 10000)
        continue;
      approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
      boundRect[i] = boundingRect(Mat(contours_poly[i]));
      Scalar color =
          Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      // drawContours( ttCopy, contours, i, color, 2, 8, hierarchy, 0, Point()
      // );
      rectangle(ttCopy, boundRect[i].tl(), boundRect[i].br(), color, 3, 8, 0);
      roi_contours.push_back(
          make_pair(timetableImage(boundRect[i]).clone(), contours_poly[i]));
    }
  }

  imshow("huha", ttCopy);
  waitKey();

  std::sort(roi_contours.begin(), roi_contours.end(), contour_sorter());
  for (size_t i = 0; i < roi_contours.size(); ++i) {
    boundRect[i] = boundingRect(Mat(roi_contours[i].second));
  }
  cout << boundRect.size() << endl;
  auto res = group_cells(roi_contours);

  tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();
  // Initialize tesseract-ocr with English, without specifying tessdata path
  if (tess->Init(NULL, "eng")) {
    fprintf(stderr, "Could not initialize tesseract.\n");
    exit(1);
  }

  int tt_left = numeric_limits<int>::max();
  int tt_right = numeric_limits<int>::lowest();
  int tt_top = numeric_limits<int>::max();
  int tt_bottom = numeric_limits<int>::lowest();
  for (size_t i = 0; i < roi_contours.size(); ++i) {
    Mat temp_gray, temp_bw;
    cvtColor(roi_contours[i].first, temp_gray, CV_BGR2GRAY);
    threshold(temp_gray, temp_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    double avg_brightness = 0;
    for (int r = 0; r < temp_bw.rows; ++r) {
      for (int c = 0; c < temp_bw.cols; ++c) {
        int pix_val = temp_bw.at<uchar>(r, c);
        avg_brightness += pix_val;
      }
    }
    avg_brightness /= (temp_bw.rows * temp_bw.cols);
    if (avg_brightness < 128)
      bitwise_not(temp_bw, temp_bw);
    fastNlMeansDenoising(temp_bw, temp_bw);
    imshow("yolo", temp_bw);
    Rect temp_bounding(boundingRect(roi_contours[i].second));
    cout << temp_bounding.x << " " << temp_bounding.y << endl;
    if (temp_bounding.x < tt_left)
      tt_left = temp_bounding.x;
    if (temp_bounding.x < tt_top)
      tt_top = temp_bounding.x;
    if (temp_bounding.y + temp_bounding.width > tt_right)
      tt_right = temp_bounding.y + temp_bounding.width;
    if (temp_bounding.y + temp_bounding.height > tt_bottom)
      tt_bottom = temp_bounding.y + temp_bounding.height;
    waitKey();
    tess->SetImage((uchar *)temp_bw.data, temp_bw.size().width,
                   temp_bw.size().height, temp_bw.channels(), temp_bw.step1());
    tess->Recognize(0);
    const char *out = tess->GetUTF8Text();
    if (strlen(out) > 3)
      cout << out << endl;
  }
  printf("%d %d %d %d\n", tt_left, tt_right, tt_top, tt_bottom);
  tess->End();
  return 0;
}
