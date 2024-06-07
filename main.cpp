#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <numeric>

using namespace cv;
using namespace std;

Mat canny(Mat img)
{
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_RGB2GRAY);
    Mat edges;
    Canny(gray_img, edges, 110, 120);
    return edges;
}

float vectorAverage(vector<float> input_vec)
{
    float average = accumulate(input_vec.begin(), input_vec.end(), 0.0) / input_vec.size();
    return average;
}

void drawLines(Mat img, vector<Vec4f> lines, int thickness = 5)
{
    Scalar right_color = Scalar(0, 0, 255);
    Scalar left_color = Scalar(0, 255, 255);
    vector<float> rightSlope, leftSlope, rightIntercept, leftIntercept;
    for (Vec4f line : lines)
    {
        float x1 = line[0];
        float y1 = line[1];
        float x2 = line[2];
        float y2 = line[3];
        float slope = (y1 - y2) / (x1 - x2);
        if (slope > 0.5)
        {
            if (x1 > 500)
            {
                float yintercept = y2 - (slope * x2);
                rightSlope.push_back(slope);
                rightIntercept.push_back(yintercept);
            }
        }
        else if (slope < -0.5)
        {
            if (x1 < 700)
            {
                float yintercept = y2 - (slope * x2);
                leftSlope.push_back(slope);
                leftIntercept.push_back(yintercept);
            }
        }
    }

    float left_intercept_avg = vectorAverage(leftIntercept);
    float right_intercept_avg = vectorAverage(rightIntercept);
    float left_slope_avg = vectorAverage(leftSlope);
    float right_slope_avg = vectorAverage(rightSlope);

    int left_line_x1 = (int)round((0.65 * img.rows - left_intercept_avg) / left_slope_avg);
    int left_line_x2 = (int)round((img.rows - left_intercept_avg) / left_slope_avg);
    int right_line_x1 = (int)round((0.65 * img.rows - right_intercept_avg) / right_slope_avg);
    int right_line_x2 = (int)round((img.rows - right_intercept_avg) / right_slope_avg);
    Point line_vertices[1][4];
    line_vertices[0][0] = Point(left_line_x1, (int)round(0.65 * img.rows));
    line_vertices[0][1] = Point(left_line_x2, img.rows);
    line_vertices[0][2] = Point(right_line_x2, img.rows);
    line_vertices[0][3] = Point(right_line_x1, (int)round(0.65 * img.rows));
    const Point *inner_shape[1] = {line_vertices[0]};
    int n_vertices[] = {4};
    int lineType = LINE_8;
    fillPoly(img, inner_shape, n_vertices, 1, Scalar(0, 50, 0), lineType);
    line(img, Point(left_line_x1, (int)round(0.65 * img.rows)), Point(left_line_x2, img.rows), left_color, 10);
    line(img, Point(right_line_x1, (int)round(0.65 * img.rows)), Point(right_line_x2, img.rows), right_color, 10);
}

Mat hough_lines(Mat img, double rho, double theta, int threshold, double min_line_len, double max_line_gap)
{
    vector<Vec4f> lines;
    Mat line_img(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));
    HoughLinesP(img, lines, rho, theta, threshold, min_line_len, max_line_gap);

    drawLines(line_img, lines);
    return line_img;
}

Mat lineDetect(Mat img)
{
    return hough_lines(img, 1, CV_PI / 180, 50, 100, 100);
}

Mat weightedImage(Mat img, Mat initialImg, double alpha = 0.8, double beta = 1.0, double gamma = 0.0)
{
    Mat weightedImg;
    addWeighted(img, alpha, initialImg, beta, gamma, weightedImg);
    return weightedImg;
}

Mat laneDetection(Mat src)
{
    Mat colorMasked, roiImg, cannyImg, houghImg, finalImg;
    Mat hls, yellowMask, whiteMask, maskN, masked;
    cvtColor(src, hls, COLOR_RGB2HLS);
    inRange(hls, Scalar(100, 0, 90), Scalar(50, 255, 255), yellowMask);
    inRange(hls, Scalar(0, 70, 0), Scalar(255, 255, 255), whiteMask);
    bitwise_or(yellowMask, whiteMask, maskN);
    bitwise_and(src, src, masked, maskN = maskN);

    int x = masked.cols;
    int y = masked.rows;
    Point polygonVertices[1][4];
    polygonVertices[0][0] = Point(0, y);
    polygonVertices[0][1] = Point(x, y);
    polygonVertices[0][2] = Point((int)round(0.55 * x), (int)round(0.6 * y));
    polygonVertices[0][3] = Point((int)round(0.45 * x), (int)round(0.6 * y));
    const Point *polygons[1] = {polygonVertices[0]};
    int n_vertices[] = {4};
    Mat mask(y, x, CV_8UC1, Scalar(0));
    int lineType = LINE_8;
    fillPoly(mask, polygons, n_vertices, 1, Scalar(255, 255, 255), lineType);
    Mat maskedImage;
    bitwise_and(masked, masked, maskedImage, mask = mask);
    cannyImg = canny(maskedImage);
    houghImg = lineDetect(cannyImg);
    finalImg = weightedImage(houghImg, src);
    return finalImg;
}

class StopSignDetector
{
private:
    CascadeClassifier cascade;

public:
    StopSignDetector()
    {
        cascade.load("./xmlfile/stop_sign_classifier_2.xml");
    }

    vector<Rect> detectStopSigns(const Mat &img)
    {
        vector<Rect> found;
        cascade.detectMultiScale(img, found);

        // Filter out regions based on position and aspect ratio
        vector<Rect> filteredRegions;
        for (const auto &region : found)
        {
            double aspectRatio = static_cast<double>(region.width) / region.height;
            // Check if the aspect ratio is close to 1
            if (aspectRatio > 0.7 && aspectRatio < 1.3)
            {
                // Check if the region is not near the top of the frame
                if (region.y > img.rows / 3)
                {
                    filteredRegions.push_back(region);
                }
            }
        }

        return filteredRegions;
    }
};

int main(int argc, char *argv[])
{
    // Check for the correct number of command line arguments
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <video_file_name> [--show] [--store <output_file_name>]" << endl;
        return -1;
    }

    // Load the video file
    VideoCapture cap(argv[1]);

    if (!cap.isOpened())
    {
        cout << "Error: Unable to open video file." << endl;
        return -1;
    }

    // Create a window for displaying intermediate results if --show flag is provided
    bool showIntermediate = false;
    string intermediateWindowName = "Intermediate Results";
    if (argc > 2 && string(argv[2]) == "--show")
    {
        showIntermediate = true;
        namedWindow(intermediateWindowName, WINDOW_AUTOSIZE);
    }

    // Create a video writer if --store flag is provided
    bool storeResults = false;
    VideoWriter outputVideo;
    string outputFileName;
    if (argc > 3 && string(argv[3]) == "--store" && argc > 4)
    {
        storeResults = true;
        outputFileName = argv[4];
        int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
        Size frameSize = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
        outputVideo.open(outputFileName, codec, cap.get(CAP_PROP_FPS), frameSize, true);
    }

    bool showNormalFrames = false;
    cout << "Press space to turn on or turn off the features." << endl;

    // Pedestrian detection
    CascadeClassifier pedestrianDetector;
    pedestrianDetector.load("./xmlfile/pedetrian1.xml");

    // Load the Haar Cascade classifier XML file for detecting cars
    CascadeClassifier carDetector;
    carDetector.load("./xmlfile/carDetection.xml");

    // Load the Haar Cascade classifier XML file for traffic light detection
    CascadeClassifier trafficLightDetector;
    trafficLightDetector.load("./xmlfile/traffic_light2.xml");

    // Framerate calculation
    auto start = std::chrono::steady_clock::now();
    int fps = 0;
    int currentFps = 1;

    // Process each frame of the video
    Mat frame;
    StopSignDetector stopSignDetector;
    while (cap.read(frame))
    {
        // Lane detection
        frame = laneDetection(frame);

        // Pedestrian detection
        vector<Rect> pedestrian;
        pedestrianDetector.detectMultiScale(frame, pedestrian, 1.1, 5);

        // Detect cars in the current frame
        vector<Rect> cars;
        carDetector.detectMultiScale(frame, cars, 1.1, 5);

        // Detect traffic light in the current frame
        vector<Rect> trafficLight;
        trafficLightDetector.detectMultiScale(frame, trafficLight, 1.1, 2);

        // Detect stop signs in the current frame
        vector<Rect> stopSigns = stopSignDetector.detectStopSigns(frame);

        // Draw stop sign circles
        for (const auto &rect : stopSigns)
        {
            Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
            int radius = max(rect.width, rect.height) / 2;
            circle(frame, center, radius, Scalar(0, 165, 255), 2);
        }

        // Draw pedestrian rectangles
        for (const auto &rect : pedestrian)
        {
            rectangle(frame, rect.tl(), rect.br(), Scalar(128, 0, 128), 2);
        }

        // Draw car rectangles
        for (const auto &rect : cars)
        {
            rectangle(frame, rect.tl(), rect.br(), Scalar(0, 255, 255), 2);
        }

        // Draw traffic light rectangles
        for (const auto &rect : trafficLight)
        {
            rectangle(frame, rect.tl(), rect.br(), Scalar(0, 0, 255), 2);
        }

        if (storeResults)
        {
            outputVideo.write(frame);
        }

        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        fps++;
        putText(frame, "Frames/second: " + to_string(currentFps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        if (duration >= 1)
        {
            currentFps = fps;
            fps = 0;
            start = std::chrono::steady_clock::now();
        }
        imshow("Object Detection", frame);
        const char key = (char)waitKey(1);
        if (key == 27 || key == 'q')
        {
            cout << "Exit requested" << endl;
            cap.release();
            if (storeResults)
                outputVideo.release();
            if (showIntermediate)
                destroyWindow(intermediateWindowName);
            return 0;
        }
        else if (key == ' ')
        {
            cout << "Data: " << showNormalFrames << endl;
            if (showNormalFrames)
            {
                showNormalFrames = 0;
                cout << "Feature Detection Started" << endl;
            }
            else
            {
                showNormalFrames = 1;
                cout << "Features detection stopped" << endl;
            }
        }
    }

    return 0;
}

