#include "YOLOutils.hpp" 

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>


// 定义默认配色方案 (BGR格式)
const std::vector<cv::Scalar> YOLOVisualizer::DEFAULT_COLORS = {
    cv::Scalar(239, 57, 136),   // #8839ef -> BGR(239, 57, 136)
    cv::Scalar(120, 138, 220),  // #dc8a78 -> BGR(120, 138, 220)
    cv::Scalar(57, 15, 210),    // #d20f39 -> BGR(57, 15, 210)
    cv::Scalar(11, 100, 254),   // #fe640b -> BGR(11, 100, 254)
    cv::Scalar(43, 160, 64),    // #40a02b -> BGR(43, 160, 64)
};

const std::vector<cv::Scalar>& YOLOVisualizer::getDefaultColors() {
    return DEFAULT_COLORS;
}

cv::Scalar YOLOVisualizer::getColorForClass(int class_idx, const std::vector<cv::Scalar>& colors) {
    const auto& color_list = colors.empty() ? DEFAULT_COLORS : colors;

    if (color_list.empty()) {
        return cv::Scalar(0, 255, 0);  // 默认绿色
    }

    // 使用模运算循环使用颜色
    return color_list[class_idx % color_list.size()];
}

void YOLOVisualizer::saveDetectResults(
    const std::string& output_path,
    const cv::Mat& img,
    const std::vector<DetectObj>& results,
    const std::vector<std::string>& class_names,
    const std::vector<cv::Scalar>& colors
) {
    cv::Mat vis_img = img.clone();
    drawDetectResults(vis_img, results, class_names, colors);

    if (!cv::imwrite(output_path, vis_img)) {
        std::cerr << "保存检测结果失败: " << output_path << std::endl;
    }
}

void YOLOVisualizer::savePoseResults(
    const std::string& output_path,
    const cv::Mat& img,
    const std::vector<PoseObj>& results,
    const int kpt_idx,
    const std::vector<std::string>& class_names,
    const std::vector<cv::Scalar>& colors,
    const cv::Scalar& kpt_color
) {
    cv::Mat vis_img = img.clone();
    drawPoseResults(vis_img, results, class_names, kpt_idx, colors, kpt_color);

    if (!cv::imwrite(output_path, vis_img)) {
        std::cerr << "保存姿态结果失败: " << output_path << std::endl;
    }
}

void YOLOVisualizer::saveSegmentResults(
    const std::string& output_path,
    const cv::Mat& img,
    const std::vector<SegmentObj>& results,
    const std::vector<std::string>& class_names,
    const std::vector<cv::Scalar>& colors,
    float alpha
) {
    cv::Mat vis_img = img.clone();
    drawSegmentResults(vis_img, results, class_names, colors, alpha);

    if (!cv::imwrite(output_path, vis_img)) {
        std::cerr << "保存分割结果失败: " << output_path << std::endl;
    }
}

void YOLOVisualizer::drawDetectResults(
    cv::Mat& img,
    const std::vector<DetectObj>& results,
    const std::vector<std::string>& class_names,
    const std::vector<cv::Scalar>& colors
) {
    for (const auto& obj : results) {
        cv::Scalar color = getColorForClass(obj.class_idx, colors);

        // 绘制边界框
        cv::rectangle(img, obj.bbox, color, 2);

        // 准备标签文本
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);

        if (obj.class_idx >= 0 && obj.class_idx < static_cast<int>(class_names.size())) {
            ss << class_names[obj.class_idx] << ": " << obj.conf;
        }
        else {
            ss << "Class" << obj.class_idx << ": " << obj.conf;
        }

        // 绘制标签
        drawLabel(img, ss.str(), cv::Point(obj.bbox.x, obj.bbox.y - 10), color);
    }
}

void YOLOVisualizer::drawPoseResults(
    cv::Mat& img,
    const std::vector<PoseObj>& results,
    const std::vector<std::string>& class_names,
    const int kpt_idx,
    const std::vector<cv::Scalar>& colors,
    const cv::Scalar& kpt_color,
    float kpt_threshold
) {
    for (const auto& obj : results) {
        cv::Scalar bbox_color = getColorForClass(obj.class_idx, colors);

        cv::rectangle(img, obj.bbox, bbox_color, 2);

        // for (size_t i = 0; i < obj.kpts.size(); ++i) {
        //     const auto& kpt = obj.kpts[i];
        //     if (kpt.z > kpt_threshold) {
        //         cv::Point2f point(kpt.x, kpt.y);
        //         cv::circle(img, point, 5, kpt_color, -1);
        //         // cv::putText(img, std::to_string(i), 
        //         //            cv::Point(static_cast<int>(point.x + 5), static_cast<int>(point.y - 5)),
        //         //            cv::FONT_HERSHEY_SIMPLEX, 0.4, kpt_color, 1);
        //     }
        // }

        const auto& kpt = obj.kpts[kpt_idx];
        if (kpt.z > kpt_threshold) {
            cv::Point2f point(kpt.x, kpt.y);
            cv::circle(img, point, 5, kpt_color, -1);
            // cv::putText(img, std::to_string(i), 
            //            cv::Point(static_cast<int>(point.x + 5), static_cast<int>(point.y - 5)),
            //            cv::FONT_HERSHEY_SIMPLEX, 0.4, kpt_color, 1);
        }

        // 准备标签文本
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);

        if (obj.class_idx >= 0 && obj.class_idx < static_cast<int>(class_names.size())) {
            ss << class_names[obj.class_idx] << ": " << obj.conf;
        }
        else {
            ss << "Class" << obj.class_idx << ": " << obj.conf;
        }

        drawLabel(img, ss.str(), cv::Point(obj.bbox.x, obj.bbox.y - 10), bbox_color);
    }
}

void YOLOVisualizer::drawSegmentResults(
    cv::Mat& img,
    const std::vector<SegmentObj>& results,
    const std::vector<std::string>& class_names,
    const std::vector<cv::Scalar>& colors,
    float alpha
) {
    cv::Mat mask_overlay = img.clone();

    for (const auto& obj : results) {
        cv::Scalar color = getColorForClass(obj.class_idx, colors);

        cv::rectangle(img, obj.bbox, color, 2);

        if (!obj.mask.empty()) {
            try {
                // 确保掩码区域在图像范围内
                cv::Rect safe_bbox = obj.bbox & cv::Rect(0, 0, img.cols, img.rows);
                if (safe_bbox.width > 0 && safe_bbox.height > 0) {
                    // 调整掩码尺寸以匹配边界框
                    cv::Mat resized_mask;
                    if (obj.mask.size() != safe_bbox.size()) {
                        cv::resize(obj.mask, resized_mask, safe_bbox.size());
                    }
                    else {
                        resized_mask = obj.mask;
                    }

                    mask_overlay(safe_bbox).setTo(color, resized_mask);
                }
            }
            catch (const cv::Exception& e) {
                std::cerr << "绘制掩码时出错: " << e.what() << std::endl;
            }
        }

        // 准备标签文本
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);

        if (obj.class_idx >= 0 && obj.class_idx < static_cast<int>(class_names.size())) {
            ss << class_names[obj.class_idx] << ": " << obj.conf;
        }
        else {
            ss << "Class" << obj.class_idx << ": " << obj.conf;
        }

        drawLabel(img, ss.str(), cv::Point(obj.bbox.x, obj.bbox.y - 10), color);
    }

    // 融合原图和掩码覆盖层
    cv::addWeighted(img, 1.0f - alpha, mask_overlay, alpha, 0, img);
}

void YOLOVisualizer::drawLabel(
    cv::Mat& img,
    const std::string& text,
    const cv::Point& position,
    const cv::Scalar& color,
    double font_scale
) {
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    int thickness = 2;
    int baseline = 0;

    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

    cv::Point label_pos = position;
    if (label_pos.y < text_size.height + 10) {
        label_pos.y = text_size.height + 10;
    }
    if (label_pos.x < 0) {
        label_pos.x = 0;
    }
    if (label_pos.x + text_size.width > img.cols) {
        label_pos.x = img.cols - text_size.width;
    }

    // 绘制文本背景
    cv::Point bg_p1(label_pos.x - 2, label_pos.y - text_size.height - baseline - 2);
    cv::Point bg_p2(label_pos.x + text_size.width + 2, label_pos.y + baseline + 2);
    cv::rectangle(img, bg_p1, bg_p2, color, -1);

    // 绘制文本 (使用白色或黑色文本，根据背景颜色自动选择)
    cv::Scalar text_color;
    double brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299);
    if (brightness > 127) {
        text_color = cv::Scalar(0, 0, 0);  // 黑色文本
    }
    else {
        text_color = cv::Scalar(255, 255, 255);  // 白色文本
    }

    cv::putText(img, text, label_pos, font_face, font_scale, text_color, thickness);
}
