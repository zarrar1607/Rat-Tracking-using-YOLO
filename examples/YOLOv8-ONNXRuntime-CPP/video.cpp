#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <algorithm> // For std::max_element
#include "inference.h"

// Function to load class names from a YAML file (assumed simple format)
int ReadCocoYaml(YOLO_V8 *&p)
{
    std::ifstream file("rat.yaml"); // Ensure the YAML file is in the current directory
    if (!file.is_open())
    {
        std::cerr << "❌ Failed to open YAML file" << std::endl;
        return 1;
    }
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }
    std::size_t start = 0, end = 0;
    // Find the section where the names are defined
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
            std::cout << lines[i] << std::endl;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string key, value;
        std::getline(ss, key, ':');
        std::getline(ss, value);
        std::cout << key << ":" << value << std::endl;
        // Remove any leading/trailing whitespace from value
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        names.push_back(value);
    }

    std::cout << "names size: " << names.size() << std::endl;
    p->classes = names;
    return 0;
}

// Function to process real-time video frames: run inference, draw only the highest-confidence detection, and display the video.
void ProcessVideo(YOLO_V8 *&p, const std::string &videoFile)
{
    // // Open the default camera (device 0)
    // cv::VideoCapture cap(0);
    // if (!cap.isOpened())
    // {
    //     std::cerr << "Error opening video stream" << std::endl;
    //     return;
    // }

    cv::VideoCapture cap(videoFile);
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) // Fallback to a default value if FPS is not valid
        fps = 30.0;
    cv::VideoWriter video("output.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          fps,
                          cv::Size(frame_width, frame_height));
    if (!video.isOpened())
    {
        std::cerr << "Error: Could not open the video writer." << std::endl;
        return;
    }
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video file: " << videoFile << std::endl;
        return;
    }

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<DL_RESULT> results;
        // Run inference on the current frame
        p->RunSession(frame, results);

        // If any detections exist, select the one with the highest confidence
        if (!results.empty())
        {
            auto bestDetection = std::max_element(results.begin(), results.end(),
                                                  [](const DL_RESULT &a, const DL_RESULT &b)
                                                  {
                                                      return a.confidence < b.confidence;
                                                  });
            if (bestDetection != results.end())
            {
                const DL_RESULT &r = *bestDetection;
                // Draw the bounding box
                cv::rectangle(frame, r.box, cv::Scalar(0, 255, 0), 2);
                // Create the label text (e.g., "class_name 0.85")
                std::string label = p->classes[r.classId] + " " + std::to_string(r.confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                int top = std::max(r.box.y, labelSize.height);
                // Draw a filled rectangle to serve as background for the label text
                cv::rectangle(frame, cv::Point(r.box.x, top - labelSize.height),
                              cv::Point(r.box.x + labelSize.width, top + baseLine),
                              cv::Scalar(0, 255, 0), cv::FILLED);
                // Put the label text over the background rectangle
                cv::putText(frame, label, cv::Point(r.box.x, top),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
        }

        // Display the processed frame
        cv::imshow("Real-Time Object Detection", frame);
        // video.write(frame);

        // Break the loop if 'q' or ESC is pressed
        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q')
            break;
    }
    video.release();
    cap.release();
    cv::destroyAllWindows();
}

int main()
{
    // Create an instance of the YOLO_V8 detector
    YOLO_V8 *yoloDetector = new YOLO_V8;
    std::string model_path = "../best.onnx"; // Adjust the model path if needed
    std::string video_path = "../../../Video/Baseline.mp4";

    // Load the class names from the YAML file
    if (ReadCocoYaml(yoloDetector) != 0)
    {
        delete yoloDetector;
        return -1;
    }

    // Set up model parameters
    DL_INIT_PARAM params;
    params.modelPath = model_path;
    params.imgSize = {416, 416}; // Adjust the input size if necessary
    // Set thresholds as needed for your application
    params.rectConfidenceThreshold = 0.01;
    params.iouThreshold = 0.01;
    params.cudaEnable = false;
    params.modelType = YOLO_DETECT_V8;

    // Create the ONNX inference session
    yoloDetector->CreateSession(params);

    // Process real-time video from the default camera
    ProcessVideo(yoloDetector, video_path);

    delete yoloDetector;
    return 0;
}
