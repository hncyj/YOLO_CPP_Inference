#include "YOLOPipeline.hpp"

#include <opencv2/opencv.hpp>
#include <optional>

// Detect RoI Method
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
    roi_(roi)
{
    this->engine_type_ = engine_type.value_or(InferenceEngineFactory::getEngineFromModelPath(this->model_path_));
    initializeEngine();
}


void YOLOPipeline::initializeEngine() {
    this->inference_engine_ = InferenceEngineFactory::createInferenceEngine(this->engine_type_);
    
    if (!this->loadModel()) { throw std::runtime_error("Failed to load model: " + model_path_); }
    
    auto input_shape = this->inference_engine_->getInputShape();
    this->target_size_ = cv::Size(input_shape[1], input_shape[2]); // H, W

    this->yolo_preprocessor_ = YOLOPreProcessor(this->target_size_);

    // Get number of outputs and set corresponding parameters based on task type
    auto output_shapes = this->inference_engine_->getOutputShapes();
    if (output_shapes.size() == 1 && (task_type_ == YOLOTaskType::DETECT || task_type_ == YOLOTaskType::POSE)) { 
        if (task_type_ == YOLOTaskType::POSE) {
            this->num_keypoints_ = (output_shapes[0][1] - 4 - this->cls_nums_) / 3;
        }
    } else if (output_shapes.size() == 2 && task_type_ == YOLOTaskType::SEGMENT) { 
        this->mask_channels_ = output_shapes[1][1];
        this->mask_size_ = cv::Size(output_shapes[1][2], output_shapes[1][3]);

    } else {
        throw std::runtime_error("Task output size doesn't consistent with model outputs");
    }

    switch (task_type_) {
        case YOLOTaskType::DETECT:
            postprocessor_ = PostProcessorFactory::createPostProcessor(
                task_type_, config_, cls_nums_
            );
            break;
            
        case YOLOTaskType::POSE:
            postprocessor_ = PostProcessorFactory::createPostProcessor(
                task_type_, config_, cls_nums_, num_keypoints_, kpt_thresh_
            );
            break;
            
        case YOLOTaskType::SEGMENT:
            postprocessor_ = PostProcessorFactory::createPostProcessor(
                task_type_, config_, cls_nums_, 1, 0.5f, mask_channels_, mask_size_, ratio_
            );
            break;
            
        default:
            throw std::invalid_argument("Unsupported task type");
    }
}

bool YOLOPipeline::loadModel() {
    if (engine_type_ == InferenceEngine::OPENVINO) {
        auto* ov_engine = dynamic_cast<OpenVINOInference*>(inference_engine_.get());
        return ov_engine->loadModel(model_path_, device_, cache_dir_);
    } else if (engine_type_ == InferenceEngine::ONNXRUNTIME) {
        auto* ort_engine = dynamic_cast<ONNXRuntimeInference*>(inference_engine_.get());
        return ort_engine->loadModel(model_path_);
    }
    return false;
}

bool YOLOPipeline::runInference(const cv::Mat& preprocessed_img, std::vector<cv::Mat>& outputs) {
    cv::Mat engine_input = inference_engine_->convertInput(preprocessed_img);
    return inference_engine_->infer(engine_input, outputs);
}

cv::Mat YOLOPipeline::cropImageWithRoI(const cv::Mat& image, cv::Vec2d& roi_offset) {
    roi_offset = cv::Vec2d(0, 0); // default: without offset
    
    if (roi_.empty() || roi_.width <= 0 || roi_.height <= 0 || 
        (roi_.width == image.cols && roi_.height == image.rows)) {
        return image;
    }
    
    if (roi_.width != target_size_.width || roi_.height != target_size_.height) {
        std::cerr << "RoI Size error: RoI size (" << roi_.width << "x" << roi_.height 
        << ") does not match target size (" << target_size_.width << "x" << target_size_.height 
        << "). Using entire RoI." << std::endl;
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
    // Skip transformation if RoI is empty or has no offset
    if (roi_.empty() || (roi_offset[0] == 0 && roi_offset[1] == 0)) {
        return;
    }
    
    for (auto& result : results) {
        // Transform bounding box
        result.bbox.x += static_cast<int>(roi_offset[0]);
        result.bbox.y += static_cast<int>(roi_offset[1]);
        
        // For pose detection, transform keypoints as well
        if constexpr (std::is_same_v<T, PoseObj>) {
            for (auto& kpt : result.kpts) {
                if (kpt.x >= 0 && kpt.y >= 0) { // Only transform valid keypoints
                    kpt.x += static_cast<float>(roi_offset[0]);
                    kpt.y += static_cast<float>(roi_offset[1]);
                }
            }
        }
        
        // Segmentation mask coordinates are relative to bounding box, no need for additional transformation
    }
}


std::vector<DetectObj> YOLOPipeline::detectInfer(const cv::Mat& image, bool use_NMS) {
    if (task_type_ != YOLOTaskType::DETECT) {
        throw std::runtime_error("This pipeline is not configured for DETECT task");
    }
    
    // RoI crop
    cv::Vec2d roi_offset;
    cv::Mat processed_image = cropImageWithRoI(image, roi_offset);
    
    // pre-processor
    auto preprocess_result = yolo_preprocessor_.preprocess(processed_image);

    // infer
    std::vector<cv::Mat> outputs;
    if (!runInference(preprocess_result.processed_img, outputs) || outputs.empty()) {
        return {};
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
    
    return results;
}



std::vector<PoseObj> YOLOPipeline::poseInfer(const cv::Mat& image, bool use_NMS) {
    if (task_type_ != YOLOTaskType::POSE) {
        throw std::runtime_error("This pipeline is not configured for POSE task");
    }
    
    // RoI crop
    cv::Vec2d roi_offset;
    cv::Mat processed_image = cropImageWithRoI(image, roi_offset);
    
    // pre-processor
    auto preprocess_result = yolo_preprocessor_.preprocess(processed_image);

    // infer
    std::vector<cv::Mat> outputs;
    if (!runInference(preprocess_result.processed_img, outputs) || outputs.empty()) {
        return {};
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
    
    return results;
}

std::vector<SegmentObj> YOLOPipeline::segInfer(const cv::Mat& image, bool use_NMS) {
    if (task_type_ != YOLOTaskType::SEGMENT) {
        throw std::runtime_error("This pipeline is not configured for SEGMENT task");
    }
    
    // RoI crop
    cv::Vec2d roi_offset;
    cv::Mat processed_image = cropImageWithRoI(image, roi_offset);
    
    // pre-processor
    auto preprocess_result = yolo_preprocessor_.preprocess(processed_image);
    
    // infer
    std::vector<cv::Mat> outputs;
    if (!runInference(preprocess_result.processed_img, outputs) || outputs.size() != 2) {
        return {};
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
    
    return results;
}