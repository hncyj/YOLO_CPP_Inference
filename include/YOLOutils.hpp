#pragma once

#include "YOLOProcessor.hpp"

#include <opencv2/opencv.hpp>
#include <vector>  
#include <string>
#include <chrono>
#include <map>

/**
 * @brief YOLO result visualization utility class
 */
class YOLOVisualizer {
private:
    // Color scheme (BGR format)
    static const std::vector<cv::Scalar> DEFAULT_COLORS;
    
public:
    /**
     * @brief Save detection result visualization image
     * @param output_path Output image path
     * @param img Original image
     * @param results Detection results
     * @param class_names Class name array
     * @param colors Color list (optional, uses built-in colors by default)
     */
    static void saveDetectResults(
        const std::string& output_path, 
        const cv::Mat& img, 
        const std::vector<DetectObj>& results, 
        const std::vector<std::string>& class_names,
        const std::vector<cv::Scalar>& colors = {}
    );
    
    /**
     * @brief Save pose detection result visualization image
     * @param output_path Output image path
     * @param img Original image
     * @param results Pose detection results
     * @param class_names Class name list
     * @param colors Color list (optional, uses built-in colors by default)
     * @param kpt_color Keypoint color (optional)
     */
    static void savePoseResults(
        const std::string& output_path, 
        const cv::Mat& img,
        const std::vector<PoseObj>& results,
        const int kpt_idx,
        const std::vector<std::string>& class_names,
        const std::vector<cv::Scalar>& colors = {},
        const cv::Scalar& kpt_color = cv::Scalar(57, 15, 210)
    );
    
    /**
     * @brief Save segmentation result visualization image
     * @param output_path Output image path
     * @param img Original image
     * @param results Segmentation results
     * @param class_names Class name list
     * @param colors Color list (optional, uses built-in colors by default)
     * @param alpha Mask transparency (0.0-1.0)
     */
    static void saveSegmentResults(
        const std::string& output_path, 
        const cv::Mat& img,
        const std::vector<SegmentObj>& results,
        const std::vector<std::string>& class_names,
        const std::vector<cv::Scalar>& colors = {},
        float alpha = 0.8f
    );

    /**
     * @brief Draw detection results on image (without saving)
     * @param img Image (will be modified)
     * @param results Detection results
     * @param class_names Class name list
     * @param colors Color list (optional, uses built-in colors by default)
     */
    static void drawDetectResults(
        cv::Mat& img,
        const std::vector<DetectObj>& results,
        const std::vector<std::string>& class_names,
        const std::vector<cv::Scalar>& colors = {}
    );

    /**
     * @brief Draw pose results on image (without saving)
     * @param img Image (will be modified)
     * @param results Pose detection results
     * @param class_names Class name list
     * @param colors Color list (optional, uses built-in colors by default)
     * @param kpt_color Keypoint color (optional)
     * @param kpt_threshold Keypoint confidence threshold
     */
    static void drawPoseResults(
        cv::Mat& img,
        const std::vector<PoseObj>& results,
        const std::vector<std::string>& class_names,
        const int kpt_idx,
        const std::vector<cv::Scalar>& colors = {},
        const cv::Scalar& kpt_color = cv::Scalar(239, 57, 136),
        float kpt_threshold = 0.5f
    );

    /**
     * @brief Draw segmentation results on image (without saving)
     * @param img Image (will be modified)
     * @param results Segmentation results
     * @param class_names Class name list
     * @param colors Color list (optional, uses built-in colors by default)
     * @param alpha Mask transparency
     */
    static void drawSegmentResults(
        cv::Mat& img,
        const std::vector<SegmentObj>& results,
        const std::vector<std::string>& class_names,
        const std::vector<cv::Scalar>& colors = {},
        float alpha = 0.8f
    );

    /**
     * @brief Get default color list
     * @return Default color vector
     */
    static const std::vector<cv::Scalar>& getDefaultColors();
    
    /**
     * @brief Get color for specific class
     * @param class_idx Class index
     * @param colors Color list (uses default colors if empty)
     * @return Corresponding color
     */
    static cv::Scalar getColorForClass(int class_idx, const std::vector<cv::Scalar>& colors = {});

private:
    /**
     * @brief Draw label text
     * @param img Image
     * @param text Text content
     * @param position Text position
     * @param color Text color
     * @param font_scale Font size
     */
    static void drawLabel(
        cv::Mat& img,
        const std::string& text,
        const cv::Point& position,
        const cv::Scalar& color,
        double font_scale = 0.7
    );
};
