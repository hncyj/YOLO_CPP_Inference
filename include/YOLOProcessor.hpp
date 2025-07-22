#pragma once


#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <memory>
#include <algorithm>

/**
 * @brief YOLOPipeline 状态码枚举类
 */
enum class YOLOStatusCode {
    SUCCESS = 0,
    INIT_FAILED,
    INVALID_INPUT,
    INVALID_CONFIGURATION,
    MODEL_LOAD_FAILED,
    INFERENCE_FAILED,
    UNSUPPORTED_TASK,
    MEMORY_ERROR,
    ENGINE_INIT_FAILED
};


/**
 * @brief 预处理结果结构体
 *
 * @param processed_img cv::Mat 处理后的图像数据
 * @param transfrom_params cv::Vec4d (ratio_w, ratio_h, offset_x, offset_y) 结果图像与原图的比
 * @param original_size cv::Size 原图尺寸
 * @param status_code 状态码
 * @param status_msg 状态信息
 */
struct PreProcessResult {
    cv::Mat processed_img;
    cv::Vec4d transform_params; // (ratio_w, ratio_h, offset_x, offset_y)
    cv::Size original_size;
    YOLOStatusCode status_code;
    std::string status_msg;

    bool isSuccess() const { return status_code == YOLOStatusCode::SUCCESS; }
};


class YOLOPreProcessor {
public:
    /**
     * @brief YOLOPreProcessor constructor
     *
     * @param target_size 目标尺寸, 默认640x640
     * @param auto_size 是否自动调整填充尺寸, 默认false
     * @param scale_fill 是否使用拉伸填充, 默认false
     * @param scale_up 是否允许放大小于输入尺寸的图像, 默认true
     * @param center 是否居中放置, 默认true
     * @param stride 步长参数, 模型最大下采样倍率, 默认32
     */
    YOLOPreProcessor(
        cv::Size target_size = cv::Size(640, 640),
        bool auto_size = false,
        bool scale_fill = false,
        bool scale_up = true,
        bool center = true,
        int stride = 32
    );

    /**
     * @brief 图像预处理函数
     *
     * @param img cv::Mat 输入图像
     * @return PreprocessResult 预处理结果，包含处理后图像和变换参数
     */
    PreProcessResult preprocess(const cv::Mat& img);

    /**
     * @brief 将预处理后图像中的坐标转换回原图坐标
     *
     * @param point cv::Point2f& 预处理后图像中的点坐标
     * @param transform_params cv::Vec4d& 变换参数
     * @param original_size cv::Size 原图尺寸
     * @return cv::Point2f 原图中的坐标
     */
    cv::Point2f transformCoordinate(
        const cv::Point2f& point,
        const cv::Vec4d& transform_params,
        const cv::Size& original_size
    );

private:
    cv::Size target_size_;      // 目标尺寸
    bool auto_size_;            // 是否自动调整填充尺寸
    bool scale_fill_;           // 是否使用拉伸填充
    bool scale_up_;             // 是否允许放大
    bool center_;               // 是否居中放置
    int stride_;                // 步长参数
};


/**
 * @brief YOLO模型任务类型枚举类
 */
enum class YOLOTaskType {
    DETECT,
    POSE,
    SEGMENT
};


/**
 * @brief Detect 模型后处理结果结构体
 * @param class_idx 类别id
 * @param conf 类别置信度
 * @param bbox 类别检测框: (x, y, w, h)
 */
struct DetectObj {
    int class_idx;
    float conf;
    cv::Rect bbox;

    DetectObj() = default;
    DetectObj(int idx, float conf, const cv::Rect& box) : class_idx(idx), conf(conf), bbox(box) {};
};


/**
 * @brief Pose 模型后处理结果结构体
 * @param class_idx 类别id
 * @param conf 类别置信度
 * @param bbox 类别检测框: (x, y, w, h)
 * @param kpts 关键点数组: (x, y, conf) * kpt_nums
 */
struct PoseObj : public DetectObj {
    std::vector<cv::Point3f> kpts; // array of (x, y, conf)

    PoseObj() = default;
    PoseObj(int idx, float conf, const cv::Rect& box, const std::vector<cv::Point3f>& kpts) : DetectObj(idx, conf, box), kpts(kpts) {};
};


/**
 * @brief Segment 模型后处理结果结构体
 * @param class_idx 类别id
 * @param conf 类别置信度
 * @param bbox 类别检测框: (x, y, w, h)
 * @param mask segment 模型对输入预测的分割掩码
 */
struct SegmentObj : public DetectObj {
    cv::Mat mask;

    SegmentObj() = default;
    SegmentObj(int idx, float conf, const cv::Rect& box, const cv::Mat& mask) : DetectObj(idx, conf, box), mask(mask) {};
};


/**
 * @brief YOLO 推理结果结构体，存储了结果状态码与结果数组
 */
template<typename T>
struct YOLOResult {
    YOLOStatusCode status_code = YOLOStatusCode::SUCCESS;
    std::string status_message;
    std::vector<T> data;

    YOLOResult() = default;
    YOLOResult(YOLOStatusCode code, const std::string& message, const std::vector<T>& result)
        : status_code(code), status_message(message), data(result) {
    }

    bool isSuccess() const { return status_code == YOLOStatusCode::SUCCESS; }
};


/**
 * @brief PostProcess 配置结构体
 * @param conf_thresh 类别置信度阈值
 * @param nms_thresh nms(iou) 阈值
 */
struct PostProcessConfig {
    float conf_thresh = 0.2f;      // 置信度阈值
    float nms_thresh = 0.7f;       // NMS(IoU) 阈值

    PostProcessConfig() = default;
    PostProcessConfig(float conf_thresh, float nms_thresh) : conf_thresh(conf_thresh), nms_thresh(nms_thresh) {};
};


/**
 * @brief YOLO Detection PostProcessor
 */
class DetectPostProcessor {
protected:
    PostProcessConfig config_;
    int class_nums_;

    cv::Point2f transformCoordinate(const cv::Point2f& point, const cv::Vec4d& transform_params, const cv::Size& original_size) const;
    cv::Rect bbox2Rect(float cx, float cy, float w, float h, const cv::Vec4d& transform_params, const cv::Size& original_size) const;

public:
    DetectPostProcessor(const PostProcessConfig& config, const int class_nums = 1);
    virtual ~DetectPostProcessor() = default;
    std::vector<DetectObj> decode_output(const cv::Mat& output, const cv::Vec4d& transform_params, const cv::Size& original_size, bool use_NMS);
};

/**
 * @brief YOLO Pose PostProcessor
 */
class PosePostProcessor : public DetectPostProcessor {
private:
    int num_kpts_;
    float kpts_conf_threshold_;

public:
    PosePostProcessor(const PostProcessConfig& config, const int class_nums = 1, const int num_keypoints = 1, float kpt_thresh = 0.5f);
    std::vector<PoseObj> decode_output(const cv::Mat& output, const cv::Vec4d& transform_params, const cv::Size& original_size, bool use_NMS);
};


/**
 * @brief YOLO Segment PostProcessor
 */
class SegmentPostProcessor : public DetectPostProcessor {
private:
    int mask_channels_;    // 掩码通道数, 默认值: 32
    float ratio_;          // 检测框较短边拓展比例 ratio_, 防止边界框过度切割掩码
    cv::Size mask_size_;   // 掩码尺寸, 默认值: 320 * 320

    /**
     * @brief 拓展预测框的较短边, 防止矩形框过度截断 mask
     *
     * @param rect 矩形框
     * @param origin_size 原始图像尺寸
     */
    cv::Rect expandRect(const cv::Rect& rect, const cv::Size& origin_size);

    /**
     * @brief 生成分割掩码
     *
     * @param coeffs 掩码系数
     * @param protos 原型掩码
     * @param transform_params 变换参数
     * @param original_size 原始图像尺寸
     * @param box 检测框
     * @param mask_out 输出掩码
     */
    void generateMask(
        const std::vector<float>& coeffs, const cv::Mat& protos,
        const cv::Vec4d& transform_params, const cv::Size& original_size,
        const cv::Size& target_size, const cv::Rect& box, cv::Mat& mask_out
    );

public:
    SegmentPostProcessor(const PostProcessConfig& config, const int class_nums = 1, int mask_channels = 32, const cv::Size& mask_size = cv::Size(320, 320), const float ratio = 0.2f);
    std::vector<SegmentObj> decode_output(const cv::Mat& output0, const cv::Mat& output1, const cv::Vec4d& transform_params, const cv::Size& original_size, const cv::Size& target_size, bool use_NMS);
};


/**
 * @brief 后处理器工厂类
 *
 * 当前支持 DETECT, POSE, SEGMENT 三种 YOLO 任务的后处理
 */
class PostProcessorFactory {
public:
    static std::unique_ptr<DetectPostProcessor> createPostProcessor(
        YOLOTaskType type,
        const PostProcessConfig& config,
        const int class_nums,

        YOLOStatusCode& status_code,
        std::string& status_msg,

        int num_keypoints = 1,                          // for pose
        float kpt_thresh = 0.5f,                        // for pose

        int mask_channels = 32,                         // for segment
        const cv::Size& mask_size = cv::Size(320, 320), // for segment
        float ratio = 0.2f                              // for segment
    ) {
        switch (type) {
        case YOLOTaskType::DETECT:
            return std::make_unique<DetectPostProcessor>(config, class_nums);

        case YOLOTaskType::POSE:
            return std::make_unique<PosePostProcessor>(config, class_nums, num_keypoints, kpt_thresh);

        case YOLOTaskType::SEGMENT:
            return std::make_unique<SegmentPostProcessor>(config, class_nums, mask_channels, mask_size, ratio);

        default:
            status_code = YOLOStatusCode::UNSUPPORTED_TASK;
            status_msg = "Unsupported YOLO type" + std::to_string(static_cast<int>(type));
            return nullptr;
        }
    }
};
