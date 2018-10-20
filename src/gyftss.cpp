#include <Python.h>
#include <algorithm>
#include <iostream>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#ifdef __cplusplus
extern "C" {
#endif

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

struct mat_sorter {
  bool operator()(const Mat &a, const Mat &b) {
    int aRows = a.rows;
    int aCols = a.cols;
    int bRows = b.rows;
    int bCols = b.cols;
    return (aRows*aCols) > (bRows*bCols);
  }
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
  starts.push_back(roi_contours.size());
  sort(starts.begin(), starts.end());
  return starts;
}

Mat binariseTimetable(Mat input) {
  Mat blueBack, background, titleFont, bodyFont, borders, result;
  inRange(input, Scalar(213, 122, 83), Scalar(217, 126, 87), blueBack);
  inRange(input, Scalar(252, 236, 237), Scalar(255, 240, 241), background);
  inRange(input, Scalar(253, 253, 253), Scalar(255, 255, 255), titleFont);
  inRange(input, Scalar(100, 0, 0), Scalar(104, 2, 2), bodyFont);
  inRange(input, Scalar(50, 0, 0), Scalar(60, 2, 2), borders);
  result = blueBack + background;
  return result;
}

vector<vector<string>> get_timetable(string filename) {
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
  rsz = scaled;

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
    if (area < 100) // value is randomly chosen, found by trial and error
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
  }

  // Sort rois by area
  std::sort(rois.begin(), rois.end(), mat_sorter());

  Mat timetableImage = rois[0];
  Mat ttGray, ttBw, ttCopy;
  ttCopy = timetableImage.clone();
  cvtColor(timetableImage, ttGray, CV_BGR2GRAY);
  adaptiveThreshold(~ttGray, ttBw, 255, CV_ADAPTIVE_THRESH_MEAN_C,
                    THRESH_BINARY, 15, -2);
  // ttBw = binariseTimetable(ttCopy);
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
      if (area < 800 || area > 10000)
        continue;
      approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
      boundRect[i] = boundingRect(Mat(contours_poly[i]));
      Scalar color =
          Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      rectangle(ttCopy, boundRect[i].tl(), boundRect[i].br(), color, 3, 8, 0);
      roi_contours.push_back(
          make_pair(timetableImage(boundRect[i]).clone(), contours_poly[i]));
    }
  }

  // imshow("huha", ttCopy);
  // waitKey();

  std::sort(roi_contours.begin(), roi_contours.end(), contour_sorter());
  for (size_t i = 0; i < roi_contours.size(); ++i) {
    boundRect[i] = boundingRect(Mat(roi_contours[i].second));
  }
  vector<int> starts = group_cells(roi_contours);

  tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();
  // Initialize tesseract-ocr with English, without specifying tessdata path
  if (tess->Init(NULL, "eng")) {
    fprintf(stderr, "Could not initialize tesseract.\n");
    exit(1);
  }

  vector<vector<string>> timetable;
  for (size_t i = 0; i < starts.size() - 1; ++i) {
    vector<string> timetable_row;
    for (int j = starts[i]; j < starts[i + 1]; ++j) {
      Mat temp_gray, temp_bw;
      cvtColor(roi_contours[j].first, temp_gray, CV_BGR2GRAY);
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
      // imshow("yolo", temp_bw);
      // waitKey();
      tess->SetImage((uchar *)temp_bw.data, temp_bw.size().width,
                     temp_bw.size().height, temp_bw.channels(),
                     temp_bw.step1());
      tess->Recognize(0);
      const char *out = tess->GetUTF8Text();
      if (strlen(out) > 3) {
        timetable_row.push_back(out);
        cout << out << endl;
      } else {
        timetable_row.push_back("");
      }
    }
    timetable.push_back(timetable_row);
  }
  tess->End();
  auto timetable_temp = timetable;
  for (size_t i = 0; i < timetable_temp.size(); ++i) {
    for (size_t j = 0; j < timetable_temp[i].size(); ++j) {
      for (size_t k = 0; k < timetable_temp[i][j].length(); k++) {
        if (timetable_temp[i][j][k] == '\n') {
          cout << " ";
          timetable_temp[i][j][k] = ' ';
        } else
          cout << timetable_temp[i][j][k];
      }
      cout << " || ";
    }
    cout << endl;
    cout << "===============\n";
  }
  return timetable;
}

int main(int argc, char **argv) {
  // Load source image
  if (argc != 2) {
    cerr << "Incorrect number of arguments!" << endl;
  }

  string filename(argv[1]);
  get_timetable(filename);
  return 0;
}

PyObject *vectorToTuple_String(const vector<string> &data) {
  PyObject *tuple = PyTuple_New(data.size());
  if (!tuple)
    throw logic_error("Unable to allocate memory for Python tuple");
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject *str = PyBytes_FromString(data[i].c_str());
    if (!str) {
      Py_DECREF(tuple);
      throw logic_error("Unable to allocate memory for Python tuple");
    }
    PyTuple_SET_ITEM(tuple, i, str);
  }

  return tuple;
}

PyObject *vectorVectorToTuple_String(const vector<vector<string>> &data) {
  PyObject *tuple = PyTuple_New(data.size());
  if (!tuple)
    throw logic_error("Unable to allocate memory for Python tuple");
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject *subTuple = NULL;
    try {
      subTuple = vectorToTuple_String(data[i]);
    } catch (logic_error &e) {
      throw e;
    }
    if (!subTuple) {
      Py_DECREF(tuple);
      throw logic_error("Unable to allocate memory for Python tuple of tuples");
    }
    PyTuple_SET_ITEM(tuple, i, subTuple);
  }

  return tuple;
}

static PyObject *gyftss_wrapper(PyObject *Py_UNUSED(self), PyObject *args) {
  const char *filename;
  vector<vector<string>> timetable;
  PyObject *ret;

  if (!PyArg_ParseTuple(args, "s", &filename))
    return NULL;

  timetable = get_timetable(filename);
  ret = vectorVectorToTuple_String(timetable);

  return ret;
}

static PyMethodDef GyftssMethods[] = {
    {"convert", gyftss_wrapper, METH_VARARGS,
     "Get timetable as array from screenshot"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef gyftssmodule = {PyModuleDef_HEAD_INIT, "gyftss", NULL,
                                          -1, GyftssMethods};

PyMODINIT_FUNC PyInit_gyftss(void) { return PyModule_Create(&gyftssmodule); }

#ifdef __cplusplus
}
#endif
