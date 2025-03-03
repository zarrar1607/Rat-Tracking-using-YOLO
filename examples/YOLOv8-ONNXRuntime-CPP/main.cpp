#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>

// Classifier function to detect objects and print their class names and confidence scores
void Classifier(YOLO_V8 *&p)
{
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images";

    for (const auto &entry : std::filesystem::directory_iterator(imgs_path))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            std::string img_path = entry.path().string();
            cv::Mat img = cv::imread(img_path);

            if (img.empty())
            {
                std::cerr << "⚠️ Failed to load image: " << img_path << std::endl;
                continue;
            }

            std::vector<DL_RESULT> results;
            char *ret = p->RunSession(img, results);

            std::cout << "\n🔍 Processing Image: " << img_path << std::endl;
            for (const auto &r : results)
            {
                std::cout << "✅ Detected Class: " << p->classes[r.classId] 
                          << " | Confidence: " << std::fixed << std::setprecision(2) << r.confidence << std::endl;
            }
        }
    }
}

// Function to load class names from YAML
int ReadCocoYaml(YOLO_V8 *&p)
{
    // std::ifstream file("./rat.yaml");  // Update the path if needed
    std::ifstream file("./coco.yaml");  // Update the path if needed
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
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
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
        std::string name;
        std::getline(ss, name, ':');
        std::getline(ss, name);
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}

void ClsTest()
{
    YOLO_V8 *yoloDetector = new YOLO_V8;
    // std::string model_path = "./best.onnx";  // Adjust if using another model
    std::string model_path = "./yolov8n.onnx";  // Adjust if using another model
    ReadCocoYaml(yoloDetector);

    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.01;
    params.iouThreshold = 0.01;
    params.modelPath = model_path;
    // params.imgSize = {416, 416};
    params.imgSize = {640, 640};
    params.cudaEnable = false;
    params.modelType = YOLO_DETECT_V8;

    yoloDetector->CreateSession(params);
    Classifier(yoloDetector);
}

int main()
{
    ClsTest();
}
