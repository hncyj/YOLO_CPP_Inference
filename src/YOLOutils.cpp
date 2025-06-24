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
        } else {
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
        } else {
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
                    } else {
                        resized_mask = obj.mask;
                    }
                    
                    mask_overlay(safe_bbox).setTo(color, resized_mask);
                }
            } catch (const cv::Exception& e) {
                std::cerr << "绘制掩码时出错: " << e.what() << std::endl;
            }
        }
        
        // 准备标签文本
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        
        if (obj.class_idx >= 0 && obj.class_idx < static_cast<int>(class_names.size())) {
            ss << class_names[obj.class_idx] << ": " << obj.conf;
        } else {
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
    } else {
        text_color = cv::Scalar(255, 255, 255);  // 白色文本
    }
    
    cv::putText(img, text, label_pos, font_face, font_scale, text_color, thickness);
}


// ==================== PerformanceTester Implementation ====================

// PerformanceTester::PerformanceTester() : is_timing_(false) {}

// void PerformanceTester::start(const std::string& test_name) {
//     current_test_name_ = test_name;
//     start_time_ = std::chrono::high_resolution_clock::now();
//     is_timing_ = true;
// }

// double PerformanceTester::stop() {
//     if (!is_timing_) {
//         std::cerr << "Warning: Timer not started!" << std::endl;
//         return 0.0;
//     }
    
//     end_time_ = std::chrono::high_resolution_clock::now();
//     is_timing_ = false;
    
//     double elapsed_ms = getElapsedMs();
//     addMeasurement(elapsed_ms);
    
//     return elapsed_ms;
// }

// double PerformanceTester::getElapsedMs() const {
//     if (is_timing_) {
//         auto current = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current - start_time_);
//         return duration.count() / 1000.0;
//     } else {
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
//         return duration.count() / 1000.0;
//     }
// }

// double PerformanceTester::getElapsedUs() const {
//     if (is_timing_) {
//         auto current = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current - start_time_);
//         return static_cast<double>(duration.count());
//     } else {
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
//         return static_cast<double>(duration.count());
//     }
// }

// void PerformanceTester::reset() {
//     is_timing_ = false;
//     current_test_name_.clear();
// }

// void PerformanceTester::addMeasurement(double elapsed_ms) {
//     measurements_.push_back(elapsed_ms);
// }

// double PerformanceTester::getAverageMs() const {
//     if (measurements_.empty()) return 0.0;
//     return std::accumulate(measurements_.begin(), measurements_.end(), 0.0) / measurements_.size();
// }

// double PerformanceTester::getMinMs() const {
//     if (measurements_.empty()) return 0.0;
//     return *std::min_element(measurements_.begin(), measurements_.end());
// }

// double PerformanceTester::getMaxMs() const {
//     if (measurements_.empty()) return 0.0;
//     return *std::max_element(measurements_.begin(), measurements_.end());
// }

// size_t PerformanceTester::getMeasurementCount() const {
//     return measurements_.size();
// }

// void PerformanceTester::printStatistics(const std::string& test_name) const {
//     std::string name = test_name.empty() ? current_test_name_ : test_name;
//     if (name.empty()) name = "Performance Test";
    
//     std::cout << "\n=== " << name << " Statistics ===" << std::endl;
//     std::cout << std::fixed << std::setprecision(3);
//     std::cout << "Measurements: " << getMeasurementCount() << std::endl;
    
//     if (!measurements_.empty()) {
//         std::cout << "Average: " << getAverageMs() << " ms" << std::endl;
//         std::cout << "Minimum: " << getMinMs() << " ms" << std::endl;
//         std::cout << "Maximum: " << getMaxMs() << " ms" << std::endl;
        
//         // 计算标准差
//         double avg = getAverageMs();
//         double variance = 0.0;
//         for (double measurement : measurements_) {
//             variance += (measurement - avg) * (measurement - avg);
//         }
//         variance /= measurements_.size();
//         double std_dev = std::sqrt(variance);
//         std::cout << "Std Dev: " << std_dev << " ms" << std::endl;
//     }
//     std::cout << "==================================\n" << std::endl;
// }

// void PerformanceTester::clearMeasurements() {
//     measurements_.clear();
// }

// std::string PerformanceTester::getCurrentTimestamp() {
//     auto now = std::chrono::system_clock::now();
//     auto time_t = std::chrono::system_clock::to_time_t(now);
//     auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
//         now.time_since_epoch()) % 1000;
    
//     std::stringstream ss;
//     ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
//     ss << "." << std::setfill('0') << std::setw(3) << ms.count();
//     return ss.str();
// }

// // ==================== BenchmarkTester Implementation ====================

// void BenchmarkTester::startTest(const std::string& test_name) {
//     testers_[test_name].start(test_name);
// }

// double BenchmarkTester::endTest(const std::string& test_name) {
//     auto it = testers_.find(test_name);
//     if (it != testers_.end()) {
//         return it->second.stop();
//     }
//     std::cerr << "Warning: Test '" << test_name << "' not found!" << std::endl;
//     return 0.0;
// }

// PerformanceTester& BenchmarkTester::getTester(const std::string& test_name) {
//     return testers_[test_name];
// }

// void BenchmarkTester::printAllStatistics() const {
//     std::cout << "\n======= Benchmark Results =======" << std::endl;
//     for (const auto& pair : testers_) {
//         pair.second.printStatistics(pair.first);
//     }
// }

// void BenchmarkTester::generateReport(const std::string& output_file) const {
//     std::stringstream report;
    
//     report << "Performance Benchmark Report\n";
//     report << "Generated at: " << PerformanceTester::getCurrentTimestamp() << "\n";
//     report << "========================================\n\n";
    
//     for (const auto& pair : testers_) {
//         const std::string& name = pair.first;
//         const PerformanceTester& tester = pair.second;
        
//         report << "Test: " << name << "\n";
//         report << "  Measurements: " << tester.getMeasurementCount() << "\n";
        
//         if (tester.getMeasurementCount() > 0) {
//             report << std::fixed << std::setprecision(3);
//             report << "  Average: " << tester.getAverageMs() << " ms\n";
//             report << "  Minimum: " << tester.getMinMs() << " ms\n";
//             report << "  Maximum: " << tester.getMaxMs() << " ms\n";
//         }
//         report << "\n";
//     }
    
//     if (output_file.empty()) {
//         std::cout << report.str() << std::endl;
//     } else {
//         std::ofstream file(output_file);
//         if (file.is_open()) {
//             file << report.str();
//             file.close();
//             std::cout << "Report saved to: " << output_file << std::endl;
//         } else {
//             std::cerr << "Error: Could not create report file: " << output_file << std::endl;
//             std::cout << report.str() << std::endl;
//         }
//     }
// }

// void BenchmarkTester::clearAll() {
//     testers_.clear();
// }

// std::vector<std::string> BenchmarkTester::getTestNames() const {
//     std::vector<std::string> names;
//     for (const auto& pair : testers_) {
//         names.push_back(pair.first);
//     }
//     return names;
// }

// // ==================== AutoTimer Implementation ====================

// AutoTimer::AutoTimer(const std::string& name, bool print_on_destroy)
//     : name_(name), print_on_destroy_(print_on_destroy) {
//     timer_.start(name);
// }

// AutoTimer::~AutoTimer() {
//     double elapsed = timer_.stop();
//     if (print_on_destroy_) {
//         std::cout << "[" << name_ << "] Elapsed: " 
//                   << std::fixed << std::setprecision(3) 
//                   << elapsed << " ms" << std::endl;
//     }
// }

// double AutoTimer::getElapsedMs() const {
//     return timer_.getElapsedMs();
// }

// void AutoTimer::printElapsed() const {
//     std::cout << "[" << name_ << "] Current elapsed: " 
//               << std::fixed << std::setprecision(3) 
//               << timer_.getElapsedMs() << " ms" << std::endl;
// }