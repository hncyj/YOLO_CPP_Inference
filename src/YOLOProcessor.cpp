#include "YOLOProcessor.hpp"

// ==================== YOLO PreProcessor ====================

YOLOPreProcessor::YOLOPreProcessor(
    cv::Size target_size, bool auto_size,
    bool scale_fill, bool scale_up,
    bool center, int stride
) : target_size_(target_size), auto_size_(auto_size),
scale_fill_(scale_fill), scale_up_(scale_up),
center_(center), stride_(stride) 
{
}

PreProcessResult YOLOPreProcessor::preprocess(const cv::Mat& img) {
    PreProcessResult result;

    if (img.empty()) {
        result.status_code = YOLOStatusCode::INVALID_INPUT;
        result.status_msg = "Input image is empty";

        return result;
    }

    
    double ratio_w, ratio_h;
    int new_unpad_w, new_unpad_h;
    int dw, dh;

    if (scale_fill_) {
        // 拉伸填充模式：直接缩放到目标尺寸
        ratio_w = static_cast<double>(target_size_.width) / img.cols;
        ratio_h = static_cast<double>(target_size_.height) / img.rows;
        new_unpad_w = target_size_.width;
        new_unpad_h = target_size_.height;
        dw = dh = 0;
    }
    else {
        // letterbox 填充模式：保持宽高比
        double r = std::min(static_cast<double>(target_size_.width) / img.cols, static_cast<double>(target_size_.height) / img.rows);

        if (!scale_up_) { r = std::min(r, 1.0); }

        ratio_w = ratio_h = r;

        // 计算缩放后、填充前的尺寸
        new_unpad_w = static_cast<int>(round(img.cols * r));
        new_unpad_h = static_cast<int>(round(img.rows * r));

        // 计算填充数值
        dw = target_size_.width - new_unpad_w;
        dh = target_size_.height - new_unpad_h;

        if (auto_size_) {
            dw = dw % stride_;
            dh = dh % stride_;
        }
    }

    // 计算四个方向的填充数值
    int pad_left = center_ ? dw / 2 : 0;
    int pad_top = center_ ? dh / 2 : 0;
    int pad_right = dw - pad_left;
    int pad_bottom = dh - pad_top;

    cv::Mat resized_img;
    if (img.size() != cv::Size(new_unpad_w, new_unpad_h)) {
        resize(img, resized_img, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    }
    else {
        resized_img = img.clone();
    }

    if (dw > 0 || dh > 0) {
        copyMakeBorder(
            resized_img, result.processed_img,
            pad_top, pad_bottom, pad_left, pad_right,
            cv::BORDER_CONSTANT, cv::Scalar::all(0) // 与 YOLO 训练填充颜色保持一致
        );
    }
    else {
        result.processed_img = resized_img;
    }

    result.original_size = img.size();
    result.transform_params = cv::Vec4d(ratio_w, ratio_h, pad_left, pad_top);

    if (result.processed_img.size() != target_size_) { 
        result.status_code = YOLOStatusCode::INVALID_CONFIGURATION;
        result.status_msg = "Preprocessing failed: incorrect output size";

        return result;
    }

    result.status_code = YOLOStatusCode::SUCCESS;

    return result;
}


cv::Point2f YOLOPreProcessor::transformCoordinate(
    const cv::Point2f& point,
    const cv::Vec4d& transform_params,
    const cv::Size& original_size
) {
    double ratio_w = transform_params[0];
    double ratio_h = transform_params[1];
    double pad_x = transform_params[2];
    double pad_y = transform_params[3];

    float orig_x = (point.x - pad_x) / ratio_w;
    float orig_y = (point.y - pad_y) / ratio_h;

    orig_x = std::max(0.0f, std::min(orig_x, static_cast<float>(original_size.width - 1)));
    orig_y = std::max(0.0f, std::min(orig_y, static_cast<float>(original_size.height - 1)));

    return cv::Point2f(orig_x, orig_y);
}


// ==================== YOLO PostProcessor ====================

cv::Point2f DetectPostProcessor::transformCoordinate(const cv::Point2f& point, const cv::Vec4d& transform_params, const cv::Size& original_size) const {
    double ratio_w = transform_params[0];
    double ratio_h = transform_params[1];
    double pad_x = transform_params[2];
    double pad_y = transform_params[3];

    float orig_x = (point.x - pad_x) / ratio_w;
    float orig_y = (point.y - pad_y) / ratio_h;

    // 限制在图像边界内
    orig_x = std::max(0.0f, std::min(orig_x, static_cast<float>(original_size.width - 1)));
    orig_y = std::max(0.0f, std::min(orig_y, static_cast<float>(original_size.height - 1)));

    return cv::Point2f(orig_x, orig_y);
};

cv::Rect DetectPostProcessor::bbox2Rect(float cx, float cy, float w, float h, const cv::Vec4d& transform_params, const cv::Size& original_size) const {
    double ratio_w = transform_params[0];
    double ratio_h = transform_params[1];
    double pad_x = transform_params[2];
    double pad_y = transform_params[3];

    // 直接计算原图尺度下的宽高
    float w_orig = w / ratio_w;
    float h_orig = h / ratio_h;

    int left = std::max(0, static_cast<int>((cx - pad_x) / ratio_w - w_orig / 2 + 0.5f));
    int top = std::max(0, static_cast<int>((cy - pad_y) / ratio_h - h_orig / 2 + 0.5f));
    int right = std::min(original_size.width, static_cast<int>((cx - pad_x) / ratio_w + w_orig / 2 + 0.5f));
    int bottom = std::min(original_size.height, static_cast<int>((cy - pad_y) / ratio_h + h_orig / 2 + 0.5f));

    if (right <= left || bottom <= top) {
        std::cerr << "Warning: Invalid bbox dimensions, returning zero-sized rect" << std::endl;
        return cv::Rect(0, 0, 0, 0);
    }

    return cv::Rect(left, top, right - left, bottom - top);
}


// ==================== Detect PostProcessor ====================

DetectPostProcessor::DetectPostProcessor(const PostProcessConfig& config, const int class_nums)
    : config_(config), class_nums_(class_nums)
{
}

std::vector<DetectObj> DetectPostProcessor::decode_output(const cv::Mat& output, const cv::Vec4d& transform_params, const cv::Size& original_size, bool use_NMS) {
    // output.shape: [1, bbox_nums, 4 + cls_nums]
    int cls_nums = this->class_nums_;
    int bbox_nums = output.rows;

    std::vector<DetectObj> result;
    std::vector<int> ids;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    ids.reserve(bbox_nums);
    scores.reserve(bbox_nums);
    boxes.reserve(bbox_nums);

    int data_width = 4 + cls_nums;
    float* pdata = (float*)output.data;

    // 获取候选框
    for (int r = 0; r < bbox_nums; ++r) {
        float max_cls_conf = 0.0f;
        int max_cls_conf_id = -1;
        for (int cls = 0; cls < cls_nums; ++cls) {
            float conf = pdata[4 + cls];
            if (conf > max_cls_conf) {
                max_cls_conf = conf;
                max_cls_conf_id = cls;
            }
        }

        if (max_cls_conf > this->config_.conf_thresh) {
            // (center_x, center_y, w, h) -> (tl_x, tl_y, w, h)
            cv::Rect box = bbox2Rect(pdata[0], pdata[1], pdata[2], pdata[3], transform_params, original_size);

            ids.emplace_back(max_cls_conf_id);
            scores.emplace_back(max_cls_conf);
            boxes.emplace_back(box);
        }

        pdata += data_width;
    }

    if (use_NMS) {
        std::vector<int> NMS_indices;
        cv::dnn::NMSBoxes(boxes, scores, this->config_.conf_thresh, this->config_.nms_thresh, NMS_indices);
        result.reserve(NMS_indices.size());
        for (const auto& idx : NMS_indices) {
            DetectObj obj(ids[idx], scores[idx], boxes[idx]);
            result.emplace_back(std::move(obj));
        }
    }
    else {
        // 返回置信度最高的检测框
        if (!ids.empty()) {
            auto max_iter = std::max_element(scores.begin(), scores.end());
            size_t max_idx = std::distance(scores.begin(), max_iter);
            DetectObj obj(ids[max_idx], scores[max_idx], boxes[max_idx]);
            result.emplace_back(std::move(obj));
        }
    }

    return result;
}


// ==================== Pose PostProcessor ====================

PosePostProcessor::PosePostProcessor(const PostProcessConfig& config, const int class_nums, int num_keypoints, float kpt_thresh)
    : DetectPostProcessor(config, class_nums), num_kpts_(num_keypoints), kpts_conf_threshold_(kpt_thresh)
{
}


std::vector<PoseObj> PosePostProcessor::decode_output(const cv::Mat& output, const cv::Vec4d& transform_params, const cv::Size& original_size, bool use_NMS) {
    // output.shape: [1, bbox_nums, 4 + cls_nums + 3 * kpt_nums]
    int cls_nums = this->class_nums_;
    int kpt_nums = this->num_kpts_;
    int bbox_nums = output.rows;

    std::vector<PoseObj> result;
    std::vector<int> ids;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point3f>> kpts;

    ids.reserve(bbox_nums);
    scores.reserve(bbox_nums);
    boxes.reserve(bbox_nums);
    kpts.reserve(bbox_nums);

    int data_width = 4 + cls_nums + 3 * kpt_nums;
    float* pdata = (float*)output.data;

    for (int r = 0; r < bbox_nums; ++r) {
        float max_cls_conf = 0.0f;
        int max_cls_conf_id = -1;
        for (int cls = 0; cls < cls_nums; ++cls) {
            float conf = pdata[4 + cls];
            if (conf > max_cls_conf) {
                max_cls_conf = conf;
                max_cls_conf_id = cls;
            }
        }

        if (max_cls_conf >= this->config_.conf_thresh) {
            std::vector<cv::Point3f> keypoints;
            keypoints.reserve(kpt_nums);

            for (int k = 0; k < kpt_nums; k++) {
                int pdata_base = 4 + cls_nums + k * 3;
                if (pdata_base + 2 < data_width) {
                    float kpt_x = pdata[pdata_base];
                    float kpt_y = pdata[pdata_base + 1];
                    float kpt_conf = pdata[pdata_base + 2];

                    cv::Point2f kpt_orig{ -1.0f, -1.0f };
                    if (kpt_conf >= this->kpts_conf_threshold_) {
                        kpt_orig = transformCoordinate(cv::Point2f(kpt_x, kpt_y), transform_params, original_size);
                    }

                    keypoints.emplace_back(kpt_orig.x, kpt_orig.y, kpt_conf);
                }
            }

            cv::Rect box = bbox2Rect(pdata[0], pdata[1], pdata[2], pdata[3], transform_params, original_size);

            ids.emplace_back(max_cls_conf_id);
            scores.emplace_back(max_cls_conf);
            boxes.emplace_back(box);
            kpts.emplace_back(keypoints);
        }

        pdata += data_width;
    }

    if (use_NMS) {
        std::vector<int> NMS_indices;
        cv::dnn::NMSBoxes(boxes, scores, this->config_.conf_thresh, this->config_.nms_thresh, NMS_indices);

        result.reserve(NMS_indices.size());
        for (const auto& idx : NMS_indices) {
            PoseObj obj(ids[idx], scores[idx], boxes[idx], kpts[idx]);
            result.emplace_back(std::move(obj));
        }
    }
    else {
        // 返回置信度最高的检测框
        if (!ids.empty()) {
            auto max_iter = std::max_element(scores.begin(), scores.end());
            size_t max_idx = std::distance(scores.begin(), max_iter);
            PoseObj obj(ids[max_idx], scores[max_idx], boxes[max_idx], kpts[max_idx]);
            result.emplace_back(std::move(obj));
        }
    }
    return result;
}


// ==================== Segment PostProcessor ====================

SegmentPostProcessor::SegmentPostProcessor(const PostProcessConfig& config, const int class_nums, int mask_channels, const cv::Size& mask_size, const float ratio)
    : DetectPostProcessor(config, class_nums), mask_channels_(mask_channels), mask_size_(mask_size), ratio_(ratio)
{
}


std::vector<SegmentObj> SegmentPostProcessor::decode_output(const cv::Mat& output0, const cv::Mat& output1, const cv::Vec4d& transform_params, const cv::Size& original_size, const cv::Size& target_size, bool use_NMS) {
    std::vector<SegmentObj> result;

    if (output0.empty() || output1.empty()) { return result; }

    int cls_nums = this->class_nums_;
    int data_width = 4 + cls_nums + this->mask_channels_;
    int bbox_nums = output0.rows;

    std::vector<int> class_ids;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> coefficients;

    class_ids.reserve(bbox_nums);
    scores.reserve(bbox_nums);
    boxes.reserve(bbox_nums);
    coefficients.reserve(bbox_nums);

    float* pdata = (float*)output0.data;

    for (int r = 0; r < bbox_nums; ++r) {
        cv::Mat cls_scores_mat(1, cls_nums, CV_32FC1, pdata + 4);
        cv::Point max_loc;
        double max_cls_conf;
        cv::minMaxLoc(cls_scores_mat, nullptr, &max_cls_conf, nullptr, &max_loc);

        if (max_cls_conf >= this->config_.conf_thresh) {
            std::vector<float> mask_coeff(pdata + 4 + cls_nums, pdata + data_width);
            // // 将边界框尺寸以及左上角点位变换到原图下表示
            cv::Rect box = bbox2Rect(pdata[0], pdata[1], pdata[2], pdata[3], transform_params, original_size);

            if (box.width <= 0 || box.height <= 0) {
                std::cerr << "Warning: Invalid box dimensions: " << box << ", skipping detection" << std::endl;
                continue;
            }

            class_ids.push_back(max_loc.x);
            scores.push_back(static_cast<float>(max_cls_conf));
            boxes.push_back(box);
            coefficients.push_back(mask_coeff);
        }

        pdata += data_width;
    }

    std::vector<int> nms_indices;
    if (use_NMS && !boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, scores, this->config_.conf_thresh, this->config_.nms_thresh, nms_indices);
    }
    else {
        nms_indices.resize(boxes.size());
        std::iota(nms_indices.begin(), nms_indices.end(), 0);
    }

    result.reserve(nms_indices.size());

    for (int idx : nms_indices) {
        cv::Rect origin_box = boxes[idx];
        cv::Rect expand_box = expandRect(origin_box, original_size);

        if (expand_box.width <= 0 || expand_box.height <= 0) {
            std::cerr << "Warning: Invalid expanded box: " << expand_box << ", using original" << std::endl;
            expand_box = origin_box;
        }

        SegmentObj seg_obj;
        seg_obj.class_idx = class_ids[idx];
        seg_obj.conf = scores[idx];
        seg_obj.bbox = expand_box;

        // 生成分割掩码
        cv::Mat mask;
        try {
            generateMask(coefficients[idx], output1, transform_params, original_size, target_size, expand_box, mask);
            seg_obj.mask = mask;
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Failed to generate mask for detection " << idx << ": " << e.what() << std::endl;
            seg_obj.mask = cv::Mat::zeros(expand_box.size(), CV_8UC1);
        }

        result.push_back(std::move(seg_obj));
    }

    // 合并同类激光线
    // 如果不需要则之间返回 results 即可
    if (result.size() <= 0) { return result; }

    std::vector<SegmentObj> res(class_nums_);
    for (int i = 0; i < res.size(); ++i) {
        res[i].class_idx = i;
        res[i].conf = 0.0f;
    }

    for (const auto& obj : result) {
        // 合并置信度
        auto& merged_obj = res[obj.class_idx];
        merged_obj.conf = std::max(obj.conf, merged_obj.conf);

        // 合并边界框
        if (merged_obj.bbox.area() == 0) {
            merged_obj.bbox = obj.bbox;
            merged_obj.mask = obj.mask.clone();
        }
        else {
            int x1 = std::min(merged_obj.bbox.x, obj.bbox.x);
            int y1 = std::min(merged_obj.bbox.y, obj.bbox.y);
            int x2 = std::max(merged_obj.bbox.x + merged_obj.bbox.width, obj.bbox.x + obj.bbox.width);
            int y2 = std::max(merged_obj.bbox.y + merged_obj.bbox.height, obj.bbox.y + obj.bbox.height);

            cv::Rect new_bbox(x1, y1, x2 - x1, y2 - y1);

            // 创建新的合并掩码模版
            // 由于mask是相对于各自边界框的, 所以不能直接按位与
            cv::Mat new_mask = cv::Mat::zeros(new_bbox.size(), CV_8UC1);

            cv::Rect old_roi(
                merged_obj.bbox.x - new_bbox.x,
                merged_obj.bbox.y - new_bbox.y,
                merged_obj.bbox.width,
                merged_obj.bbox.height
            );

            if (old_roi.x >= 0 && old_roi.y >= 0 && old_roi.x + old_roi.width <= new_mask.cols && old_roi.y + old_roi.height <= new_mask.rows) {
                merged_obj.mask.copyTo(new_mask(old_roi));
            }

            cv::Rect cur_roi(
                obj.bbox.x - new_bbox.x,
                obj.bbox.y - new_bbox.y,
                obj.bbox.width,
                obj.bbox.height
            );

            if (cur_roi.x >= 0 && cur_roi.y >= 0 && cur_roi.x + cur_roi.width <= new_mask.cols && cur_roi.y + cur_roi.height <= new_mask.rows) {
                cv::Mat temp_mask = new_mask(cur_roi);
                cv::bitwise_or(temp_mask, obj.mask, temp_mask);
            }

            // 更新合并结果
            merged_obj.bbox = new_bbox;
            merged_obj.mask = new_mask;
        }
    }

    result.clear();

    // 只返回有检测结果的类别
    for (const auto& obj : res) {
        if (obj.conf > 0.0f && obj.bbox.area() > 0) {
            result.push_back(obj);
        }
    }

    return result;
}


cv::Rect SegmentPostProcessor::expandRect(const cv::Rect& original_box, const cv::Size& image_size) {
    if (this->ratio_ <= 0.0) {
        return original_box;
    }

    double w = static_cast<double>(original_box.width);
    double h = static_cast<double>(original_box.height);

    cv::Rect expanded_box = original_box;

    // 拓展较短边
    if (w <= h) {
        double expanded_w = w * (1.0 + ratio_);
        double center_x = original_box.x + w / 2.0;
        int new_left = static_cast<int>(center_x - expanded_w / 2.0 + 0.5);
        expanded_box = cv::Rect(new_left, original_box.y, static_cast<int>(expanded_w + 0.5), original_box.height);
    }
    else {
        double expanded_h = h * (1.0 + ratio_);
        double center_y = original_box.y + h / 2.0;
        int new_top = static_cast<int>(center_y - expanded_h / 2.0 + 0.5);
        expanded_box = cv::Rect(original_box.x, new_top, original_box.width, static_cast<int>(expanded_h + 0.5));
    }

    return expanded_box & cv::Rect(0, 0, image_size.width, image_size.height);
}


void SegmentPostProcessor::generateMask(
    const std::vector<float>& coeffs, const cv::Mat& protos,
    const cv::Vec4d& transform_params, const cv::Size& original_size,
    const cv::Size& target_size,
    const cv::Rect& box, cv::Mat& mask_out
) {
    int seg_w = mask_size_.width;
    int seg_h = mask_size_.height;
    int net_w = target_size.width;
    int net_h = target_size.height;

    // 计算在 proto 特征图中的ROI
    int r_x = floor((box.x * transform_params[0] + transform_params[2]) / net_w * seg_w);
    int r_y = floor((box.y * transform_params[1] + transform_params[3]) / net_h * seg_h);
    int r_w = ceil(((box.x + box.width) * transform_params[0] + transform_params[2]) / net_w * seg_w) - r_x;
    int r_h = ceil(((box.y + box.height) * transform_params[1] + transform_params[3]) / net_h * seg_h) - r_y;

    // 边界检查
    r_x = std::max(0, std::min(r_x, seg_w - 1));
    r_y = std::max(0, std::min(r_y, seg_h - 1));
    r_w = std::max(1, std::min(r_w, seg_w - r_x));
    r_h = std::max(1, std::min(r_h, seg_h - r_y));

    std::vector<cv::Range> roi_ranges = { cv::Range(0, 1), cv::Range::all(), cv::Range(r_y, r_y + r_h), cv::Range(r_x, r_x + r_w) };
    cv::Mat temp_mask = protos(roi_ranges).clone();
    cv::Mat protos_reshaped = temp_mask.reshape(0, { mask_channels_, r_w * r_h });

    // 将coeffs转换为Mat并进行矩阵乘法
    cv::Mat coeffs_mat(1, coeffs.size(), CV_32F, (void*)coeffs.data());
    cv::Mat matmul_result = (coeffs_mat * protos_reshaped).t();
    cv::Mat mask_feature = matmul_result.reshape(1, { r_h, r_w });
    cv::Mat sigmoid_mask = mask_feature;

    // Sigmoid
    cv::exp(-mask_feature, sigmoid_mask);
    sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);

    // retina method
    int left = floor((net_w / static_cast<double>(seg_w) * r_x - transform_params[2]) / transform_params[0]);
    int top = floor((net_h / static_cast<double>(seg_h) * r_y - transform_params[3]) / transform_params[1]);
    int width = ceil(net_w / static_cast<double>(seg_w) * r_w / transform_params[0]);
    int height = ceil(net_h / static_cast<double>(seg_h) * r_h / transform_params[1]);

    cv::Mat resized_mask;
    cv::resize(sigmoid_mask, resized_mask, cv::Size(width, height));

    // 计算ROI IoU
    cv::Rect roi(left, top, width, height);
    cv::Rect intersection = box & roi;

    if (intersection.width <= 0 || intersection.height <= 0) {
        mask_out = cv::Mat();
        return;
    }

    // 计算在 resized_mask 中的ROI
    cv::Rect roi_in_mask(intersection.x - left, intersection.y - top,
        intersection.width, intersection.height);
    roi_in_mask = roi_in_mask & cv::Rect(0, 0, width, height);

    // 计算在输出掩码中的 ROI
    cv::Rect roi_in_bound(intersection.x - box.x, intersection.y - box.y,
        roi_in_mask.width, roi_in_mask.height);

    // 创建输出掩码
    mask_out = cv::Mat::zeros(box.size(), CV_8UC1);

    // 复制掩码数据
    if (!resized_mask.empty() && roi_in_mask.width > 0 && roi_in_mask.height > 0) {
        cv::Mat binary_mask = resized_mask(roi_in_mask) > 0.5;
        binary_mask.copyTo(mask_out(roi_in_bound));
    }
}
