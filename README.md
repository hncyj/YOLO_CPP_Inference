# YOLO Pipeline Documentation

## Overview

YOLOPipeline is a C++-based YOLO model inference framework that supports three task types: object detection, pose estimation, and image segmentation. This framework provides a complete pipeline including preprocessing, inference, and postprocessing, supports OpenVINO inference engine (OnnxRunTime Later), and offers flexible Region of Interest (RoI) processing capabilities.

## Features

- **Multi-task Support**: Supports detection, pose estimation, and segmentation YOLO tasks
- **Multiple Inference Engines**: Supports OpenVINO inference engine
- **RoI Processing**: Supports specifying regions of interest for local inference
- **Flexible Configuration**: Configurable postprocessing parameters and model parameters

## Interface Class

### YOLOPipeline

```cpp
class YOLOPipeline {
public:
    // Constructor
    YOLOPipeline(
        const std::string& model_path,
        YOLOTaskType task_type,
        const int class_nums,
        const cv::Rect& roi = cv::Rect(),
        const PostProcessConfig& config = PostProcessConfig(),
        const std::string& device = "CPU",
        const std::string& cache_dir = "model_compile_cache",
        std::optional<InferenceEngine> engine_type = InferenceEngine::OPENVINO
    );

    // Inference methods
    std::vector<DetectObj> detectInfer(const cv::Mat& image, bool use_NMS = false);
    std::vector<PoseObj> poseInfer(const cv::Mat& image, bool use_NMS = false);
    std::vector<SegmentObj> segInfer(const cv::Mat& image, bool use_NMS = true);

    // Status check
    bool isLoaded() const;
};
```

## Constructor Parameters

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| `model_path` | `std::string` | ONNX model file path | Required |
| `task_type` | `YOLOTaskType` | Task type (DETECT/POSE/SEGMENT) | Required |
| `class_nums` | `int` | Number of classes | Required |
| `roi` | `cv::Rect` | Region of interest (x, y, width, height) crop in image. | Empty (process entire image) |
| `config` | `PostProcessConfig` | Postprocessing configuration (confidence threshold, NMS threshold) | Default configuration |
| `device` | `std::string` | Inference device ("CPU", "GPU") | "CPU" |
| `cache_dir` | `std::string` | Model compilation cache directory | "model_compile_cache" |
| `engine_type` | `InferenceEngine` | Inference engine type | OPENVINO |

## Task Types

```cpp
enum class YOLOTaskType {
    DETECT,   // Object detection
    POSE,     // Pose estimation  
    SEGMENT   // Instance segmentation
};
```

### 1. Object Detection (DETECT)
Used for detecting target objects in images, returns bounding boxes and confidence scores.

### 2. Pose Estimation (POSE)
Detects keypoints, returns keypoint coordinates and confidence scores.

### 3. Instance Segmentation (SEGMENT)
Pixel-level segmentation, returns target masks and bounding boxes.

## Postprocessing Configuration

```cpp
// Example configuration
PostProcessConfig config(
    0.4f,  // Confidence threshold
    0.7f   // NMS IoU threshold
);
```


### RoI Usage Example

```cpp
cv::Rect roi(100, 100, 640, 640);

YOLOPipeline pipeline(
    model_path,
    YOLOTaskType::DETECT,
    class_nums,
    roi,  // RoI configuration
    config
);
```

## Complete Usage Example

### Usage Example

```cpp
#include "YOLOPipeline.hpp"
#include "YOLOutils.hpp"
#include <iostream>
#include <filesystem>

int main() {
    std::cout << "=== YOLO Pipeline Test ===" << std::endl;
    
    // Create output directory
    if (!std::filesystem::exists("../results")) { std::filesystem::create_directories("../results"); }

    const std::string test_image = "../resources/images/2025_06_06_09_10_29_275_crop_0035.jpg";

    try {
        const std::string model_path = "../models/seg/triseg20250520.onnx";

        PostProcessConfig config(0.1f, 0.7f);
        
        cv::Rect roi = cv::Rect(540, 765, 640, 640);

        const int class_nums = 1;
        const std::vector<std::string> class_names = { "seg_results" };

        const auto TASK = YOLOTaskType::SEGMENT;
     
        const float kpt_thresh = 0.5f;

        std::string device = "CPU";
        std::string cache_dir = "model_compile_cache";

        bool use_NMS = true;
        
        // Test OpenVINO
        std::cout << "\n=== OpenVINO Test ===" << std::endl;
        {   
            // Initialize YOLOPipeline
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
            
            // Select inference method based on task type and get inference results
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
                        auto results = pipeline.segInfer(img, true); // segmentation must use NMS
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
```

## Return Result Types

> **Note**: The current implementation has modified the segmentation results returned by the model. Elements in the original `vector<SegmentObj> result` have been merged by class, with the final result containing only the number of elements equal to the number of classes.

```cpp
/**
 * @brief Detect model postprocessing result structure
 * @param class_idx Class ID
 * @param conf Class confidence score
 * @param bbox Class bounding box: (x, y, w, h)
 */
struct DetectObj {
    int class_idx;
    float conf;
    cv::Rect bbox;

    DetectObj() = default;
    DetectObj(int idx, float conf, const cv::Rect& box) : class_idx(idx), conf(conf), bbox(box) {};
};


/**
 * @brief Pose model postprocessing result structure
 * @param class_idx Class ID
 * @param conf Class confidence score
 * @param bbox Class bounding box: (x, y, w, h)
 * @param kpts Keypoint array: (x, y, conf) * kpt_nums
 */
struct PoseObj : public DetectObj {
    std::vector<cv::Point3f> kpts; // array of (x, y, conf)

    PoseObj() = default;
    PoseObj(int idx, float conf, const cv::Rect& box, const std::vector<cv::Point3f>& kpts) : DetectObj(idx, conf, box), kpts(kpts) {};
};


/**
 * @brief Segment model postprocessing result structure
 * @param class_idx Class ID
 * @param conf Class confidence score
 * @param bbox Class bounding box: (x, y, w, h)
 * @param mask Segmentation mask predicted by the segment model for the input
 */
struct SegmentObj : public DetectObj {
    cv::Mat mask;

    SegmentObj() = default;
    SegmentObj(int idx, float conf, const cv::Rect& box, const cv::Mat& mask) : DetectObj(idx, conf, box), mask(mask) {};
};
```

## Important Notes

1. **Thread Safety**: The current implementation does not guarantee thread safety; additional synchronization is required for multi-threaded usage
2. **Memory Management**: Ensure input images are valid and avoid passing empty images
3. **Model Compatibility**: Ensure model format matches the task type


## Dependencies

- OpenCV 4.10.0
- OpenVINO Runtime
- C++17 or higher