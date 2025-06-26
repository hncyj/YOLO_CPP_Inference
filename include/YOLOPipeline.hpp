#pragma once

#include "YOLOProcessor.hpp"
#include "ModelLoader.hpp"

#include <memory>
#include <optional>
#include <stdexcept>

/**
 * @brief YOLO Pipeline Main class
 */
class YOLOPipeline {
private:
    // Model params
    std::string model_path_;
    std::string device_;
    std::string cache_dir_;

    // for Detect, Pose, Segment
    int cls_nums_;

    // POSE
    int num_keypoints_ = 1;
    float kpt_thresh_ = 0.5f;

    // SEGMENT
    int mask_channels_ = 32;
    float ratio_ = 0.2f;
    cv::Size mask_size_ = cv::Size(320, 320);
    
    
    // model task type
    YOLOTaskType task_type_;
    InferenceEngine engine_type_;

    // image roi
    cv::Rect roi_;

    // YOLO preprocessor params
    cv::Size target_size_;

    // infer engine type
    std::unique_ptr<ModelInferBase> inference_engine_;
    
    // pre post processor
    YOLOPreProcessor yolo_preprocessor_;
    std::unique_ptr<DetectPostProcessor> postprocessor_;
    
    // YOLO postprocessor params
    PostProcessConfig config_;

public:
    /**
     * @brief Detect with RoI
     */
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

    ~YOLOPipeline() = default;

    // Inference methods
    std::vector<DetectObj> detectInfer(const cv::Mat& image, bool use_NMS = false);
    std::vector<PoseObj> poseInfer(const cv::Mat& image, bool use_NMS = false);
    std::vector<SegmentObj> segInfer(const cv::Mat& image, bool use_NMS = true);

    // Set ROI
    void setRoI(const cv::Rect& roi) { this->roi_ = roi; };
    
    // Get model loading status
    bool isLoaded() const { return inference_engine_ && inference_engine_->isLoaded(); }

    // Get model input size
    cv::Size getModelInputSize() { return this->target_size_; }

private:
    /**
     * @brief Initialize inference engine and post-processor
     */
    void initializeEngine();
    
    /**
     * @brief Load model
     */
    bool loadModel();
    
    /**
     * @brief Run inference
     */
    bool runInference(const cv::Mat& preprocessed_img, std::vector<cv::Mat>& outputs);

    /**
     * @brief Crop image with RoI
     * @param image Input image
     * @param roi_offset Output RoI offset (offset_x, offset_y) for coordinate transformation in post-processing
     * @return Cropped image
     */
    cv::Mat cropImageWithRoI(const cv::Mat& image, cv::Vec2d& roi_offset);

    /**
     * @brief Transform detection results from RoI coordinates to original image coordinates
     * @param results Detection results
     * @param roi_offset RoI offset (offset_x, offset_y) for coordinate transformation
     */
    template<typename T>
    void transformResultsToOriginal(std::vector<T>& results, const cv::Vec2d& roi_offset);
};
