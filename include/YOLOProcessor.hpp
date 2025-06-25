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
 * @brief Preprocessing result structure
 * 
 * @param processed_img cv::Mat Processed image data
 * @param transfrom_params cv::Vec4d (ratio_w, ratio_h, offset_x, offset_y) Ratio between result image and original image
 * @param original_size cv::Size Original image size
 */
struct PreProcessResult {
    cv::Mat processed_img;
    cv::Vec4d transform_params; // (ratio_w, ratio_h, offset_x, offset_y)
    cv::Size original_size;
};


class YOLOPreProcessor {
public:
    /**
     * @brief YOLOPreProcessor constructor
     * 
     * @param target_size Target size, default 640x640
     * @param auto_size Whether to automatically adjust padding size, default false
     * @param scale_fill Whether to use stretch filling, default false
     * @param scale_up Whether to allow upscaling of images smaller than input size, default true
     * @param center Whether to center the image, default true
     * @param stride Stride parameter, model's maximum downsampling ratio, default 32
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
     * @brief Image preprocessing function
     * 
     * @param img cv::Mat Input image
     * @return PreprocessResult Preprocessing result, including processed image and transform parameters
     */
    PreProcessResult preprocess(const cv::Mat& img);
    
    /**
     * @brief Transform coordinates from preprocessed image back to original image
     * 
     * @param point cv::Point2f& Point coordinates in preprocessed image
     * @param transform_params cv::Vec4d& Transform parameters
     * @param original_size cv::Size Original image size
     * @return cv::Point2f Coordinates in original image
     */
    cv::Point2f transformCoordinate(
        const cv::Point2f& point, 
        const cv::Vec4d& transform_params, 
        const cv::Size& original_size
    );

private:
    cv::Size target_size_;      // Target size
    bool auto_size_;            // Whether to auto-adjust padding size
    bool scale_fill_;           // Whether to use stretch filling
    bool scale_up_;             // Whether to allow upscaling
    bool center_;               // Whether to center the image
    int stride_;                // Stride parameter
};


/**
 * @brief YOLO model task type enumeration
 */
enum class YOLOTaskType {
    DETECT,
    POSE, 
    SEGMENT
};


/**
 * @brief Detection model post-processing result structure
 * @param class_idx Class ID
 * @param conf Class confidence
 * @param bbox Bounding box: (x, y, w, h)
 */
struct DetectObj {
    int class_idx;
    float conf;
    cv::Rect bbox;

    DetectObj() = default;
    DetectObj(int idx, float conf, const cv::Rect& box): class_idx(idx), conf(conf), bbox(box) {};
};


/**
 * @brief Pose model post-processing result structure
 * @param class_idx Class ID
 * @param conf Class confidence
 * @param bbox Bounding box: (x, y, w, h)
 * @param kpts Keypoints array: (x, y, conf) * kpt_nums
 */
struct PoseObj : public DetectObj {
    std::vector<cv::Point3f> kpts; // array of (x, y, conf)

    PoseObj() = default;
    PoseObj(int idx, float conf, const cv::Rect& box, const std::vector<cv::Point3f>& kpts): DetectObj(idx, conf, box), kpts(kpts) {};
};


/**
 * @brief Segmentation model post-processing result structure
 * @param class_idx Class ID
 * @param conf Class confidence
 * @param bbox Bounding box: (x, y, w, h)
 * @param mask Segmentation mask predicted by the model
 */
struct SegmentObj : public DetectObj {
    cv::Mat mask;

    SegmentObj() = default;
    SegmentObj(int idx, float conf, const cv::Rect& box, const cv::Mat& mask): DetectObj(idx, conf, box), mask(mask) {};
};


/**
 * @brief PostProcess configuration structure
 * @param conf_thresh Confidence threshold
 * @param nms_thresh NMS(IoU) threshold
 */
struct PostProcessConfig {
    float conf_thresh = 0.2f;      // Confidence threshold
    float nms_thresh = 0.7f;       // NMS(IoU) threshold
    
    PostProcessConfig() = default;
    PostProcessConfig(float conf_thresh, float nms_thresh): conf_thresh(conf_thresh), nms_thresh(nms_thresh) {};
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
    int mask_channels_;    // Number of mask channels, default: 32
    float ratio_;          // Expansion ratio for shorter side of detection box to prevent excessive mask cropping
    cv::Size mask_size_;   // Mask size, default: 320 * 320

    /**
     * @brief Expand the shorter side of prediction box to prevent excessive mask cropping
     * 
     * @param rect Rectangle box
     * @param origin_size Original image size
     */
    cv::Rect expandRect(const cv::Rect& rect, const cv::Size& origin_size);
    
    /**
     * @brief Generate segmentation mask
     * 
     * @param coeffs Mask coefficients
     * @param protos Prototype masks
     * @param transform_params Transform parameters
     * @param original_size Original image size
     * @param box Detection box
     * @param mask_out Output mask
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
 * @brief Post-processor factory class
 * 
 * Supports post-processing for three tasks: DETECT, POSE, SEGMENT
 */
class PostProcessorFactory {
public:
    static std::unique_ptr<DetectPostProcessor> createPostProcessor(
        YOLOTaskType type, 
        const PostProcessConfig& config, 
        const int class_nums,
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
                throw std::invalid_argument("Unsupported YOLO type");
        }
    }
};