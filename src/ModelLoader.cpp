#include "ModelLoader.hpp"

#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <string>
#include <codecvt>

using namespace cv;
using namespace std;

// ==================== ONNXRuntimeInference ====================

ONNXRuntimeInference::ONNXRuntimeInference() : is_loaded_(false), status_code_(YOLOStatusCode::SUCCESS) {
}

ONNXRuntimeInference::~ONNXRuntimeInference() {
}

YOLOStatusCode ONNXRuntimeInference::initialize() {
    try {
        this->env_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "onnx_inference");
        this->status_code_ = YOLOStatusCode::SUCCESS;
        return YOLOStatusCode::SUCCESS;
    }
    catch (const std::exception& e) {
        this->status_code_ = YOLOStatusCode::ENGINE_INIT_FAILED;
        this->status_msg_ = "Failed to initialize ORT inference engine: " + std::string(e.what());
        return this->status_code_;
    }
}

cv::Mat ONNXRuntimeInference::convertInput(const cv::Mat& preprocessed_img) {
    cv::Size input_size = preprocessed_img.size();
    return cv::dnn::blobFromImage(preprocessed_img, 1 / 255.0, input_size, cv::Scalar(0, 0, 0), true, false);
}

bool ONNXRuntimeInference::loadModel(const std::string& model_path) {
    if (!this->env_) {
        this->status_code_ = YOLOStatusCode::ENGINE_INIT_FAILED;
        this->status_msg_ = "ORT inference engine not initialized";
        return false;
    }

    try {
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(0);

        // Create session
        // std::wstring model_path_w(model_path.begin(), model_path.end()); // 使用 L 前缀定义宽字符字符串
        const char* path = model_path.c_str();
        session_ = std::make_unique<Ort::Session>(*env_, path, session_options);
      

        // Get input information
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input names
        input_names_.clear();
        input_names_str_.clear();
        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_.push_back(std::string(input_name.get()));
        }
        for (const auto& name : input_names_str_) {
            input_names_.push_back(name.c_str());
        }

        // Get output names
        output_names_.clear();
        output_names_str_.clear();
        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(std::string(output_name.get()));  // 复制字符串内容
        }
        for (const auto& name : output_names_str_) {
            output_names_.push_back(name.c_str());
        }

        // Get input shape
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();

        // Get output shape array
        output_shape_.clear();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_type_info = session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto shape = output_tensor_info.GetShape();
            output_shape_.push_back(shape);
        }

        this->is_loaded_ = true;
        this->status_code_ = YOLOStatusCode::SUCCESS;
        this->status_msg_.clear();

        std::cout << "ONNX model loaded successfully: " << model_path << std::endl;
        
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        return true;
    }
    catch (const std::exception& e) {
        this->status_code_ = YOLOStatusCode::MODEL_LOAD_FAILED;
        this->status_msg_ = "Failed to load onnx model: " + std::string(e.what());
        this->is_loaded_ = false;
        return false;
    }
}

bool ONNXRuntimeInference::infer(const cv::Mat& input_blob, std::vector<cv::Mat>& outputs) {
    if (!this->is_loaded_ || !this->session_) {
        this->status_code_ = YOLOStatusCode::MODEL_LOAD_FAILED;
        this->status_msg_ = "Model not loaded.";
        return false;
    }

    try {
        outputs.clear();
        // Create input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            (float*)input_blob.data,
            input_blob.total(),
            input_shape_.data(),
            input_shape_.size()
        );

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{ nullptr },
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            output_names_.size()
        );

        // Convert output tensors to cv::Mat
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& output_tensor = output_tensors[i];
            auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            float* output_data = output_tensor.GetTensorMutableData<float>();

            if (shape.size() == 3) {
                // For 3D tensors: [batch, height, width] -> transpose to [height, width]
                Mat output_mat(Size((int)shape[2], (int)shape[1]), CV_32F, output_data);
                outputs.push_back(output_mat.t());
            }
            else if (shape.size() == 4) {
                // For 4D tensors: [batch, channels, height, width]
                vector<int> mat_shape = { 1, (int)shape[1], (int)shape[2], (int)shape[3] };
                Mat output_mat(mat_shape, CV_32F, output_data);
                outputs.push_back(output_mat);
            }
            else {
                // Default case: flatten to 2D
                int total_elements = 1;
                for (size_t j = 1; j < shape.size(); ++j) {
                    total_elements *= shape[j];
                }
                Mat output_mat(Size(total_elements, (int)shape[0]), CV_32F, output_data);
                outputs.push_back(output_mat);
            }
        }

        this->status_code_ = YOLOStatusCode::SUCCESS;
        this->status_msg_.clear();

        return true;

    } 
    catch (const std::exception& e) {
        this->status_code_ = YOLOStatusCode::INFERENCE_FAILED;
        this->status_msg_ = "ONNX inference failed: " + std::string(e.what());

        return false;
    }
}

std::vector<int64_t> ONNXRuntimeInference::getInputShape() const {
    return input_shape_;
}

std::vector<std::vector<int64_t>> ONNXRuntimeInference::getOutputShapes() const {
    return output_shape_;
};

bool ONNXRuntimeInference::isLoaded() const {
    return is_loaded_;
}

// ==================== OpenVINO Inference ====================

OpenVINOInference::OpenVINOInference() : is_loaded_(false), status_code_(YOLOStatusCode::SUCCESS) {
}

OpenVINOInference::~OpenVINOInference() {
}

YOLOStatusCode OpenVINOInference::initialize() {
    this->status_code_ = YOLOStatusCode::SUCCESS;
    return YOLOStatusCode::SUCCESS;
}

cv::Mat OpenVINOInference::convertInput(const cv::Mat& preprocessed_img) {
    cv::Mat input;
    preprocessed_img.convertTo(input, CV_32F, 1.0 / 255.0);
    return input;
}

bool OpenVINOInference::loadModel(
    const std::string& model_path,
    const std::string& device,
    const std::string& cache_dir
) {
    try {
        // Read model
        std::shared_ptr<ov::Model> model = core_.read_model(model_path);

        // Set up preprocessing
        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
        ppp.input().tensor().set_layout("NHWC");
        ppp.input().model().set_layout("NCHW");
        model = ppp.build();

        // Compile model
        ov::AnyMap compile_params;
        if (!cache_dir.empty()) {
            compile_params = { {"CACHE_DIR", cache_dir} };
        }

        compiled_model_ = core_.compile_model(model, device, compile_params);

        // Get input shape
        auto input_shape = compiled_model_.input().get_shape();
        input_shape_.clear();
        for (auto dim : input_shape) {
            input_shape_.push_back(static_cast<int64_t>(dim));
        }

        // Get output shape
        output_shape_.clear();
        auto outputs = compiled_model_.outputs();
        for (const auto& output : outputs) {
            auto shape = output.get_shape();
            std::vector<int64_t> output_dims;
            for (auto dim : shape) {
                output_dims.push_back(static_cast<int64_t>(dim));
            }
            output_shape_.push_back(output_dims);
        }

        is_loaded_ = true;

        return true;

    }
    catch (const std::exception& e) {
        std::cerr << "Failed to load OpenVINO model: " << e.what() << std::endl;
        is_loaded_ = false;
        return false;
    }
}

bool OpenVINOInference::infer(const cv::Mat& input_blob, std::vector<cv::Mat>& outputs) {
    if (!is_loaded_) {
        std::cerr << "Model not loaded" << std::endl;
        return false;
    }

    try {
        outputs.clear();

        // Create inference request
        ov::InferRequest infer_request = compiled_model_.create_infer_request();
        // Create input tensor
        ov::Tensor input_tensor(
            compiled_model_.input().get_element_type(),
            compiled_model_.input().get_shape(),
            input_blob.data
        );

        // Set input tensor and run inference
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        // Get output tensors
        for (size_t i = 0; i < compiled_model_.outputs().size(); ++i) {
            ov::Tensor output_tensor = infer_request.get_output_tensor(i);
            auto shape = output_tensor.get_shape();

            if (shape.size() == 3) {
                Mat output_mat(Size((int)shape[2], (int)shape[1]), CV_32F, output_tensor.data());
                Mat transposed = output_mat.t();
                outputs.push_back(transposed.clone());
            }
            else if (shape.size() == 4) {
                vector<int> mat_shape = { 1, (int)shape[1], (int)shape[2], (int)shape[3] };
                Mat output_mat(mat_shape, CV_32F, output_tensor.data());
                outputs.push_back(output_mat.clone());
            }
            else {
                // Default case
                int total_elements = 1;
                for (size_t j = 1; j < shape.size(); ++j) {
                    total_elements *= shape[j];
                }
                Mat output_mat(Size(total_elements, (int)shape[0]), CV_32F, output_tensor.data());
                outputs.push_back(output_mat);
            }
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "OpenVINO inference failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int64_t> OpenVINOInference::getInputShape() const {
    return input_shape_;
}

std::vector<std::vector<int64_t>> OpenVINOInference::getOutputShapes() const {
    return output_shape_;
};

bool OpenVINOInference::isLoaded() const {
    return is_loaded_;
}

std::unique_ptr<ModelInferBase> InferenceEngineFactory::createInferenceEngine(
    InferenceEngine engine_type,
    YOLOStatusCode& status_code,
    std::string& status_msg
) {
    switch (engine_type) {
        case InferenceEngine::OPENVINO: {
            auto engine = std::make_unique<OpenVINOInference>();
            status_code = engine->initialize();
            if (status_code != YOLOStatusCode::SUCCESS) {
                status_msg = engine->getStatusMsg();
                return nullptr;
            }
            return engine;
        }
        case InferenceEngine::ONNXRUNTIME: {
            auto engine = std::make_unique<ONNXRuntimeInference>();
            status_code = engine->initialize();
            if (status_code != YOLOStatusCode::SUCCESS) {
                status_msg = engine->getStatusMsg();
                return nullptr;
            }
            return engine;
        }
        default:
            status_code = YOLOStatusCode::UNSUPPORTED_TASK;
            status_msg = "Unsupported inference engine type";

            return nullptr;
        }
}
