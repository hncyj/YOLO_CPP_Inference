#pragma once

#include "YOLOProcessor.hpp"

#include <opencv2/opencv.hpp>
#include <vector>  
#include <string>
#include <chrono>
#include <map>

/**
 * @brief YOLO结果可视化工具类
 */
class YOLOVisualizer {
private:
    // 配色方案 (BGR格式)
    static const std::vector<cv::Scalar> DEFAULT_COLORS;

public:
    /**
     * @brief 保存检测结果可视化图像
     * @param output_path 输出图像路径
     * @param img 原始图像
     * @param results 检测结果
     * @param class_names 类别名称数组
     * @param colors 颜色列表（可选，默认使用内置配色）
     */
    static void saveDetectResults(
        const std::string& output_path,
        const cv::Mat& img,
        const std::vector<DetectObj>& results,
        const std::vector<std::string>& class_names,
        const std::vector<cv::Scalar>& colors = {}
    );

    /**
     * @brief 保存姿态检测结果可视化图像
     * @param output_path 输出图像路径
     * @param img 原始图像
     * @param results 姿态检测结果
     * @param class_names 类别名称列表
     * @param colors 颜色列表（可选，默认使用内置配色）
     * @param kpt_color 关键点颜色（可选）
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
     * @brief 保存分割结果可视化图像
     * @param output_path 输出图像路径
     * @param img 原始图像
     * @param results 分割结果
     * @param class_names 类别名称列表
     * @param colors 颜色列表（可选，默认使用内置配色）
     * @param alpha 掩码透明度 (0.0-1.0)
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
     * @brief 绘制检测结果到图像上（不保存）
     * @param img 图像（会被修改）
     * @param results 检测结果
     * @param class_names 类别名称列表
     * @param colors 颜色列表（可选，默认使用内置配色）
     */
    static void drawDetectResults(
        cv::Mat& img,
        const std::vector<DetectObj>& results,
        const std::vector<std::string>& class_names,
        const std::vector<cv::Scalar>& colors = {}
    );

    /**
     * @brief 绘制姿态结果到图像上（不保存）
     * @param img 图像（会被修改）
     * @param results 姿态检测结果
     * @param class_names 类别名称列表
     * @param colors 颜色列表（可选，默认使用内置配色）
     * @param kpt_color 关键点颜色（可选）
     * @param kpt_threshold 关键点置信度阈值
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
     * @brief 绘制分割结果到图像上（不保存）
     * @param img 图像（会被修改）
     * @param results 分割结果
     * @param class_names 类别名称列表
     * @param colors 颜色列表（可选，默认使用内置配色）
     * @param alpha 掩码透明度
     */
    static void drawSegmentResults(
        cv::Mat& img,
        const std::vector<SegmentObj>& results,
        const std::vector<std::string>& class_names,
        const std::vector<cv::Scalar>& colors = {},
        float alpha = 0.8f
    );

    /**
     * @brief 获取默认颜色列表
     * @return 默认颜色向量
     */
    static const std::vector<cv::Scalar>& getDefaultColors();

    /**
     * @brief 为指定类别获取颜色
     * @param class_idx 类别索引
     * @param colors 颜色列表（如果为空则使用默认颜色）
     * @return 对应的颜色
     */
    static cv::Scalar getColorForClass(int class_idx, const std::vector<cv::Scalar>& colors = {});

private:
    /**
     * @brief 绘制标签文本
     * @param img 图像
     * @param text 文本内容
     * @param position 文本位置
     * @param color 文本颜色
     * @param font_scale 字体大小
     */
    static void drawLabel(
        cv::Mat& img,
        const std::string& text,
        const cv::Point& position,
        const cv::Scalar& color,
        double font_scale = 0.7
    );
};
