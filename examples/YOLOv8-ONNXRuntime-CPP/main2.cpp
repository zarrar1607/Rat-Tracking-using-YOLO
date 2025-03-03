#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <algorithm> // For std::max_element
#include "inference.h"

// Function to load class names from a YAML file (assumed to be in a simple format)
int ReadCocoYaml(YOLO_V8*& p)
{
    std::ifstream file("rat.yaml");  // Ensure the YAML file is in the current directory
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

// Function that processes each image: runs inference, draws only the highest-confidence bounding box with label, and displays the result.
void Classifier(YOLO_V8*& p)
{
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path imagesPath = currentPath / "images";

    for (const auto &entry : std::filesystem::directory_iterator(imagesPath))
    {
        // Process only JPG and PNG files
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            std::string imgPath = entry.path().string();
            cv::Mat image = cv::imread(imgPath);

            if (image.empty())
            {
                std::cerr << "⚠️ Failed to load image: " << imgPath << std::endl;
                continue;
            }

            std::vector<DL_RESULT> results;
            // Run the model inference on the image
            p->RunSession(image, results);

            // Check if any detections were made
            if (!results.empty())
            {
                // Select the detection with the highest confidence
                auto bestDetection = std::max_element(results.begin(), results.end(),
                    [](const DL_RESULT& a, const DL_RESULT& b) {
                        return a.confidence < b.confidence;
                    });

                if (bestDetection != results.end())
                {
                    const DL_RESULT &r = *bestDetection;
                    // Draw rectangle for the bounding box
                    cv::rectangle(image, r.box, cv::Scalar(0, 255, 0), 2);

                    // Create the label text (e.g., "class_name 0.85")
                    std::string label = p->classes[r.classId] + " " + std::to_string(r.confidence);
                    int baseLine = 0;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    // Ensure the label rectangle is drawn within the image bounds
                    int top = std::max(r.box.y, labelSize.height);
                    cv::rectangle(image, cv::Point(r.box.x, top - labelSize.height),
                                  cv::Point(r.box.x + labelSize.width, top + baseLine),
                                  cv::Scalar(0, 255, 0), cv::FILLED);
                    // Put the label text on the image
                    cv::putText(image, label, cv::Point(r.box.x, top),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                }
            }
            else
            {
                std::cout << "No detections for image: " << imgPath << std::endl;
            }

            // Display the image with the best detection
            cv::imshow("Object Detection", image);
            // Wait for a key press before processing the next image
            cv::waitKey(0);
        }
    }
}

int main()
{
    // Create an instance of the YOLO_V8 detector
    YOLO_V8 *yoloDetector = new YOLO_V8;
    std::string model_path = "./best.onnx";  // Adjust the model path if needed

    // Load the class names from the YAML file
    if (ReadCocoYaml(yoloDetector) != 0)
    {
        delete yoloDetector;
        return -1;
    }

    // Set up model parameters
    DL_INIT_PARAM params;
    params.modelPath = model_path;
    params.imgSize = {416, 416};
    // Adjust the following thresholds as needed for your application
    params.rectConfidenceThreshold = 0.01;
    params.iouThreshold = 0.01;
    params.cudaEnable = false;
    params.modelType = YOLO_DETECT_V8;

    // Create the ONNX inference session
    yoloDetector->CreateSession(params);

    // Process images: run inference, annotate only the highest-confidence detection, and display
    Classifier(yoloDetector);

    delete yoloDetector;
    return 0;
}
