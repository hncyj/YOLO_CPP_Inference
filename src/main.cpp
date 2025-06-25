#include "YOLOPipeline.hpp"
#include "YOLOutils.hpp"
#include <iostream>
#include <filesystem>

int main() {
    std::cout << "=== YOLO Pipeline Test ===" << std::endl;
    
    // Create output directory
    if (!std::filesystem::exists("../results")) { std::filesystem::create_directories("../results"); }

    const std::string test_image = "../resources/images20250530/22_original.jpg";

    try {
        const std::string model_path = "../models/pose/fd0.6_320.onnx";

        PostProcessConfig config(0.1f, 0.7f);

        // Additional RoI setup
        cv::Rect roi = cv::Rect(450, 760, 200, 200);

        const int class_nums = 2;
        const std::vector<std::string> class_names = { "weldseam", "triweldseam" };

        const auto TASK = YOLOTaskType::POSE;
     
        const float kpt_thresh = 0.5f;

        std::string device = "CPU";
        std::string cache_dir = "model_compile_cache";

        bool use_NMS = true;
        
        // Test OpenVINO
        std::cout << "\n=== OpenVINO Test ===" << std::endl;
        {   
            YOLOPipeline pipeline(
                model_path,
                TASK,
                class_nums,
                roi,
                config,
                device,
                cache_dir,
                InferenceEngine::OPENVINO
            );
            
            if (pipeline.isLoaded()) {
                cv::Mat img = cv::imread(test_image);
                
                switch (TASK) {
                    case YOLOTaskType::DETECT: {
                        auto results = pipeline.detectInfer(img, use_NMS); // whether use NMS
                        std::cout << "Detected: " << results.size() << " objects" << std::endl;
                        // YOLOVisualizer::saveDetectResults("../results/ov_detect.jpg", img, results, class_names);
                        break;
                    }
                    case YOLOTaskType::POSE: {
                        auto results = pipeline.poseInfer(img, use_NMS); // whether use NMS
                        std::cout << "POSE Detected: " << results.size() << " poses" << std::endl;
                        YOLOVisualizer::savePoseResults("../results/ov_pose.jpg", img, results, 1, class_names);
                        break;
                    }
                    case YOLOTaskType::SEGMENT: {
                        auto results = pipeline.segInfer(img, true); // NMS must be used for segmentation
                        std::cout << "Segmented: " << results.size() << " objects" << std::endl;
                        YOLOVisualizer::saveSegmentResults("../results/ov_segment.jpg", img, results, class_names);
                        break;
                    }
                }
            } else {
                std::cout << "OpenVINO pipeline failed to load" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}