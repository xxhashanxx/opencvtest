#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string getDirection(float x, float y) {
    string direction;

    if (abs(y) > abs(x)) {
        if (y > 0) {
            direction = "Down";
        } else {
            direction = "Up";
        }
    } else {
        if (x > 0) {
            direction = "Right";
        } else {
            direction = "Left";
        }
    }
    return direction;
}

int main() {
    // Load the two images
    Mat img1 = imread("./image/1image.png"); // First image with small box
    Mat img2 = imread("./image/2image.png"); // Second image with the small box moved
    if (img1.empty() || img2.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    // Convert the images to grayscale
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // Threshold the images to isolate the black box
    Mat thresh1, thresh2;
    threshold(gray1, thresh1, 50, 255, THRESH_BINARY_INV);
    threshold(gray2, thresh2, 50, 255, THRESH_BINARY_INV);

    // Find contours in both images
    vector<vector<Point>> contours1, contours2;
    findContours(thresh1, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(thresh2, contours2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours1.empty() || contours2.empty()) {
        cout << "No black box detected!" << endl;
        return -1;
    }

    // Get bounding rectangles around the black box in both images
    Rect box1 = boundingRect(contours1[0]);
    Rect box2 = boundingRect(contours2[0]);

    // Calculate the center of the bounding boxes
    Point center1(box1.x + box1.width / 2, box1.y + box1.height / 2);
    Point center2(box2.x + box2.width / 2, box2.y + box2.height / 2);

    // Create ORB detector and find keypoints and descriptors
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    orb->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);

    // Match the descriptors
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("Matches", img_matches);

    // Calculate the average displacement
    Point2f avgDisplacement(0, 0);
    int count = 0;

    for (const auto& match : matches) {
        Point2f pt1 = keypoints1[match.queryIdx].pt;
        Point2f pt2 = keypoints2[match.trainIdx].pt;

        avgDisplacement += (pt2 - pt1);
        count++;
    }

    if (count > 0) {
        avgDisplacement /= count; // Average displacement
    }

    // Output the average displacement
    cout << "The black box moved by (avg): (" << avgDisplacement.x << ", " << avgDisplacement.y << ")" << endl;

    // Determine direction of movement
    string direction = getDirection(avgDisplacement.x, avgDisplacement.y);
    cout << "Direction of movement: " << direction << endl;

    // Draw arrows to indicate movement
    // Convert avgDisplacement to integer Point for drawing
    Point displacement(static_cast<int>(avgDisplacement.x), static_cast<int>(avgDisplacement.y));
    arrowedLine(img1, center1, center1 + displacement, Scalar(0, 0, 255), 2, LINE_AA);
    arrowedLine(img2, center2, center2 + displacement, Scalar(0, 0, 255), 2, LINE_AA);

    // Visualize the results
    rectangle(img1, box1, Scalar(0, 255, 0), 2);
    rectangle(img2, box2, Scalar(0, 255, 0), 2);

    // Add text to show average displacement and direction in images
    string text = "Avg Displacement: (" + to_string(static_cast<int>(avgDisplacement.x)) + ", " + to_string(static_cast<int>(avgDisplacement.y)) + ")";
    putText(img1, text + " | Direction: " + direction, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    putText(img2, text + " | Direction: " + direction, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    // Display the images
    imshow("Image 1 - Detected Box", img1);
    imshow("Image 2 - Detected Box", img2);
    waitKey(0);

    return 0;
}
