#include "YOLOPipeline.hpp"

#include <opencv2/opencv.hpp>

#include <optional>


YOLOPipeline::YOLOPipeline(
    const std::string& model_path,
    const YOLOTaskType task_type,
    const int class_nums,
    const cv::Rect& roi,
    const PostProcessConfig& config,
    const std::string& device,
    const std::string& cache_dir,
    const std::optional<InferenceEngine> engine_type
) : model_path_(model_path),
    task_type_(task_type),
    cls_nums_(class_nums),
    config_(config),
    device_(device),
    cache_dir_(cache_dir),
    roi_(roi),
    status_code_(YOLOStatusCode::SUCCESS),
    is_init_(false)
{
    this->engine_type_ = engine_type.value_or(InferenceEngineFactory::getEngineFromModelPath(this->model_path_));
    this->is_init_ = initializeEngine();
}


bool YOLOPipeline::initializeEngine() {
    YOLOStatusCode status_code = YOLOStatusCode::SUCCESS;
    std::string status_msg;

    this->inference_engine_ = InferenceEngineFactory::createInferenceEngine(this->engine_type_, status_code, status_msg);

    if (!this->inference_engine_) {
        this->status_code_ = status_code;
        this->status_msg_ = status_msg;
        
        return false;
    }

    if (!this->loadModel()) {
        this->status_code_ = YOLOStatusCode::MODEL_LOAD_FAILED;
        this->status_msg_ = "Failed to load model: " + this->model_path_;

        return false;
    }

    auto input_shape = this->inference_engine_->getInputShape();
    this->target_size_ = cv::Size(input_shape[2], input_shape[1]); // H, W

    this->yolo_preprocessor_ = YOLOPreProcessor(this->target_size_);

    // 获取模型输出张量数, 根据任务类型对相应参数进行赋值
    auto output_shapes = this->inference_engine_->getOutputShapes();
    if (output_shapes.size() == 1 && task_type_ == YOLOTaskType::DETECT) {
    }
    else if (output_shapes.size() == 1 && task_type_ == YOLOTaskType::POSE) {
        this->num_keypoints_ = (output_shapes[0][1] - 4 - this->cls_nums_) / 3;
    }
    else if (output_shapes.size() == 2 && task_type_ == YOLOTaskType::SEGMENT) {
        this->mask_channels_ = output_shapes[1][1];
        this->mask_size_ = cv::Size(output_shapes[1][2], output_shapes[1][3]);

    }
    else {
        this->status_code_ = YOLOStatusCode::INVALID_CONFIGURATION;
        this->status_msg_ = "Task output size doesn't match what the task type expects.";
        return false;
    }

    switch (task_type_) {
        case YOLOTaskType::DETECT:
            postprocessor_ = PostProcessorFactory::createPostProcessor(
                task_type_, config_, cls_nums_, status_code, status_msg
            );
            break;

        case YOLOTaskType::POSE:
            postprocessor_ = PostProcessorFactory::createPostProcessor(
                task_type_, config_, cls_nums_, status_code, status_msg, num_keypoints_, kpt_thresh_
            );
            break;

        case YOLOTaskType::SEGMENT:
            postprocessor_ = PostProcessorFactory::createPostProcessor(
                task_type_, config_, cls_nums_, status_code, status_msg, 1, 0.5f, mask_channels_, mask_size_, ratio_
            );
            break;

        default:
            this->status_code_ = YOLOStatusCode::UNSUPPORTED_TASK;
            this->status_msg_ = "Unsupported YOLO task type.";
    }

    if (!this->postprocessor_ || status_code != YOLOStatusCode::SUCCESS) {
        this->status_code_ = status_code;
        this->status_msg_ = status_msg;
        return false; 
    }

    return true;
}

bool YOLOPipeline::loadModel() {
    if (engine_type_ == InferenceEngine::OPENVINO) {
        auto* ov_engine = dynamic_cast<OpenVINOInference*>(inference_engine_.get());
        return ov_engine->loadModel(model_path_, device_, cache_dir_);
    }
    else if (engine_type_ == InferenceEngine::ONNXRUNTIME) {
        auto* ort_engine = dynamic_cast<ONNXRuntimeInference*>(inference_engine_.get());
        return ort_engine->loadModel(model_path_);
    }
    return false;
}

bool YOLOPipeline::runInference(const cv::Mat& preprocessed_img, std::vector<cv::Mat>& outputs) {
    cv::Mat engine_input = inference_engine_->convertInput(preprocessed_img);
    return inference_engine_->infer(engine_input, outputs);
}

cv::Mat YOLOPipeline::cropImageWithRoI(const cv::Mat& image, cv::Vec2d& roi_offset, YOLOStatusCode& status_code, std::string& status_msg) {
    roi_offset = cv::Vec2d(0, 0); // 默认无偏移

    // 如果RoI为空、大小为0或与图像尺寸相同，则处理输入原始图像
    if (roi_.empty() || roi_.width <= 0 || roi_.height <= 0 ||
        (roi_.width == image.cols && roi_.height == image.rows)) {
        return image;
    }

    if (roi_.width != target_size_.width || roi_.height != target_size_.height) {
        status_code = YOLOStatusCode::INVALID_INPUT;
        status_msg = "RoI Size error: RoI size dose not match target size.";
        
        return image;
    }

    int crop_x = std::max(0, std::min(roi_.x, image.cols - roi_.width));
    int crop_y = std::max(0, std::min(roi_.y, image.rows - roi_.height));

    int crop_width = std::min(roi_.width, image.cols - crop_x);
    int crop_height = std::min(roi_.height, image.rows - crop_y);

    roi_offset = cv::Vec2d(crop_x, crop_y);

    cv::Rect crop_rect(crop_x, crop_y, crop_width, crop_height);
    cv::Mat cropped = image(crop_rect).clone();

    return cropped;
}


template<typename T>
void YOLOPipeline::transformResultsToOriginal(std::vector<T>& results, const cv::Vec2d& roi_offset) {
    // 如果RoI为空或无偏移，无需变换
    if (roi_.empty() || (roi_offset[0] == 0 && roi_offset[1] == 0)) {
        return;
    }

    for (auto& result : results) {
        // 变换边界框
        result.bbox.x += static_cast<int>(roi_offset[0]);
        result.bbox.y += static_cast<int>(roi_offset[1]);

        // 如果是姿态检测，还需要变换关键点
        if constexpr (std::is_same_v<T, PoseObj>) {
            for (auto& kpt : result.kpts) {
                if (kpt.x >= 0 && kpt.y >= 0) { // 只变换有效关键点
                    kpt.x += static_cast<float>(roi_offset[0]);
                    kpt.y += static_cast<float>(roi_offset[1]);
                }
            }
        }

        // 分割掩码坐标已经是相对于边界框的，无需额外变换
    }
}


YOLOResult<DetectObj> YOLOPipeline::detectInfer(const cv::Mat& image, bool use_NMS) {
    if (!this->is_init_) {
        this->status_code_ = YOLOStatusCode::INIT_FAILED;
        this->status_msg_ = "Pipeline not properly initialized";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<DetectObj> {});
    }

    if (task_type_ != YOLOTaskType::DETECT) {
        this->status_code_ = YOLOStatusCode::UNSUPPORTED_TASK;
        this->status_msg_ = "This pipeline is not configured for DETECT task";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<DetectObj> {});
    }

    // RoI crop
    YOLOStatusCode status_code = YOLOStatusCode::SUCCESS;
    std::string status_msg;
    cv::Vec2d roi_offset;
    cv::Mat processed_image = cropImageWithRoI(image, roi_offset, status_code, status_msg);

    if (status_code != YOLOStatusCode::SUCCESS) {
        this->status_code_ = status_code;
        this->status_msg_ = status_msg;

        return YOLOResult(this->status_code_, this->status_msg_, std::vector<DetectObj> {});
    }

    // pre-processor
    auto preprocess_result = yolo_preprocessor_.preprocess(processed_image);

    if (!preprocess_result.isSuccess()) {
        this->status_code_ = preprocess_result.status_code;
        this->status_msg_ = preprocess_result.status_msg;
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<DetectObj> {});
    }

    // infer
    std::vector<cv::Mat> outputs;
    if (!runInference(preprocess_result.processed_img, outputs) || outputs.empty()) {
        this->status_code_ = YOLOStatusCode::INFERENCE_FAILED;
        this->status_msg_ = "Inference failed or output is empty.";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<DetectObj> {});
    }

    // post-processor
    auto* detect_processor = dynamic_cast<DetectPostProcessor*>(postprocessor_.get());
    auto results = detect_processor->decode_output(
        outputs[0],
        preprocess_result.transform_params,
        preprocess_result.original_size,
        use_NMS
    );

    transformResultsToOriginal(results, roi_offset);

    return YOLOResult(this->status_code_, this->status_msg_, results);
}



YOLOResult<PoseObj> YOLOPipeline::poseInfer(const cv::Mat& image, bool use_NMS) {
    if (!this->is_init_) {
        this->status_code_ = YOLOStatusCode::INIT_FAILED;
        this->status_msg_ = "Pipeline not properly initialized";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<PoseObj>{});
    }

    if (task_type_ != YOLOTaskType::POSE) {
        this->status_code_ = YOLOStatusCode::UNSUPPORTED_TASK;
        this->status_msg_ = "This pipeline is not configured for POSE task";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<PoseObj>{});
    }

    // RoI crop
    YOLOStatusCode status_code = YOLOStatusCode::SUCCESS;
    std::string status_msg;
    cv::Vec2d roi_offset;
    cv::Mat processed_image = cropImageWithRoI(image, roi_offset, status_code, status_msg);

    if (status_code != YOLOStatusCode::SUCCESS) {
        this->status_code_ = status_code;
        this->status_msg_ = status_msg;
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<PoseObj> {});
    }

    // pre-processor
    auto preprocess_result = yolo_preprocessor_.preprocess(processed_image);

    if (!preprocess_result.isSuccess()) {
        this->status_code_ = preprocess_result.status_code;
        this->status_msg_ = preprocess_result.status_msg;
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<PoseObj> {});
    }

    // infer
    std::vector<cv::Mat> outputs;
    if (!runInference(preprocess_result.processed_img, outputs) || outputs.empty()) {
        this->status_code_ = YOLOStatusCode::INFERENCE_FAILED;
        this->status_msg_ = "Inference failed or output is empty.";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<PoseObj> {});
    }

    // post-processor
    auto* pose_processor = dynamic_cast<PosePostProcessor*>(postprocessor_.get());
    auto results = pose_processor->decode_output(
        outputs[0],
        preprocess_result.transform_params,
        preprocess_result.original_size,
        use_NMS
    );

    transformResultsToOriginal(results, roi_offset);

    return YOLOResult(this->status_code_, this->status_msg_, results);
}

YOLOResult<SegmentObj> YOLOPipeline::segInfer(const cv::Mat& image, bool use_NMS) {
    if (!this->is_init_) {
        this->status_code_ = YOLOStatusCode::INIT_FAILED;
        this->status_msg_ = "Pipeline not properly initialized";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<SegmentObj>{});
    }

    if (task_type_ != YOLOTaskType::SEGMENT) {
        this->status_code_ = YOLOStatusCode::UNSUPPORTED_TASK;
        this->status_msg_ = "This pipeline is not configured for SEGMENT task";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<SegmentObj>{});
    }

    YOLOStatusCode status_code = YOLOStatusCode::SUCCESS;
    std::string status_msg;
    cv::Vec2d roi_offset;
    cv::Mat processed_image = cropImageWithRoI(image, roi_offset, status_code, status_msg);

    if (status_code != YOLOStatusCode::SUCCESS) {
        this->status_code_ = status_code;
        this->status_msg_ = status_msg;
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<SegmentObj> {});
    }

    // pre-processor
    auto preprocess_result = yolo_preprocessor_.preprocess(processed_image);

    if (!preprocess_result.isSuccess()) {
        this->status_code_ = preprocess_result.status_code;
        this->status_msg_ = preprocess_result.status_msg;
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<SegmentObj> {});
    }

    // infer
    std::vector<cv::Mat> outputs;
    if (!runInference(preprocess_result.processed_img, outputs) || outputs.empty() || outputs.size() != 2) {
        this->status_code_ = YOLOStatusCode::INFERENCE_FAILED;
        this->status_msg_ = "Inference failed or output is empty or output size not equal 2.";
        return YOLOResult(this->status_code_, this->status_msg_, std::vector<SegmentObj> {});
    }

    // post-processor
    auto* segment_processor = dynamic_cast<SegmentPostProcessor*>(postprocessor_.get());
    auto results = segment_processor->decode_output(
        outputs[0], outputs[1],
        preprocess_result.transform_params,
        preprocess_result.original_size,
        this->target_size_,
        use_NMS
    );

    transformResultsToOriginal(results, roi_offset);

    return YOLOResult(this->status_code_, this->status_msg_, results);
}
