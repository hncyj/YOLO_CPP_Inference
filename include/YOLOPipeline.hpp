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
    
    // pre post processsor
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

    // 推理方法
    std::vector<DetectObj> detectInfer(const cv::Mat& image, bool use_NMS = false);
    std::vector<PoseObj> poseInfer(const cv::Mat& image, bool use_NMS = false);
    std::vector<SegmentObj> segInfer(const cv::Mat& image, bool use_NMS = true);
    
    // 获取模型加载信息
    bool isLoaded() const { return inference_engine_ && inference_engine_->isLoaded(); }

private:
    /**
     * @brief 初始化推理引擎和后处理器
     */
    void initializeEngine();
    
    /**
     * @brief 加载模型
     */
    bool loadModel();
    
    /**
     * @brief 调用推理
     */
    bool runInference(const cv::Mat& preprocessed_img, std::vector<cv::Mat>& outputs);

    /**
     * @brief 根据RoI设置裁剪图像
     * @param image 输入图像
     * @param roi_offset 输出RoI偏移量 (offset_x, offset_y) 用于后处理坐标变换
     * @return 裁剪后的图像
     */
    cv::Mat cropImageWithRoI(const cv::Mat& image, cv::Vec2d& roi_offset);

    /**
     * @brief 将RoI坐标系下的结果转换回原图坐标系
     * @param results 检测结果
     * @param roi_offset 输出RoI偏移量 (offset_x, offset_y) 用于后处理坐标变换
     */
    template<typename T>
    void transformResultsToOriginal(std::vector<T>& results, const cv::Vec2d& roi_offset);
};
