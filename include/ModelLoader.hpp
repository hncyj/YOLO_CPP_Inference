#pragma once

#include "YOLOProcessor.hpp"

#include <string>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <openvino/openvino.hpp>

/**
 * @brief Inference engine type enumeration
 */
enum class InferenceEngine {
    OPENVINO,
    ONNXRUNTIME
};

/**
 * @brief Abstract base class for model inference
 */
class ModelInferBase {
public:
    virtual ~ModelInferBase() = default;

    /**
     * @brief 
     * @param preprocessed_img img after preprocessed
     * @return cv::Mat img after convert
     */
    virtual cv::Mat convertInput(const cv::Mat& preprocessed_img) = 0;
    
    /**
     * @brief Run inference on input image
     * @param input_blob Preprocessed input blob
     * @param outputs Vector to store output objs
     * @return true if inference successful, false otherwise
     */
    virtual bool infer(const cv::Mat& input_blob, std::vector<cv::Mat>& outputs) = 0;
    
    /**
     * @brief Get input shape of the model
     * @return Input shape as vector [batch, channels, height, width]
     */
    virtual std::vector<int64_t> getInputShape() const = 0;

    /**
     * @brief Get output shapes of the model
     * @return Vector of output shapes, each shape as vector of dimensions
     */
    virtual std::vector<std::vector<int64_t>> getOutputShapes() const = 0;
    
    /**
     * @brief Check if model is loaded successfully
     * @return true if loaded, false otherwise
     */
    virtual bool isLoaded() const = 0;
};


/**
 * @brief Inference engine factory class
 */
class InferenceEngineFactory {
public:
    /**
     * @brief Get inference engine type from model path
     * @param model_path Path to the model file
     * @return Inference engine type
     */
    static InferenceEngine getEngineFromModelPath(const std::string& model_path) {
        size_t dot_pos = model_path.find_last_of('.');
        
        if (dot_pos != std::string::npos) {
            std::string extension = model_path.substr(dot_pos);
            if (extension == ".onnx") {
                return InferenceEngine::ONNXRUNTIME;
            } else if (extension == ".xml" || extension == ".bin") {
                return InferenceEngine::OPENVINO;
            }
        }

        return InferenceEngine::OPENVINO;
    }
    
    /**
     * @brief Create inference engine instance
     * @param engine_type Engine type
     * @return Unique pointer to inference engine
     */
    static std::unique_ptr<ModelInferBase> createInferenceEngine(InferenceEngine engine_type);
};

/**
 * @brief ONNX Runtime inference implementation
 */
class ONNXRuntimeInference : public ModelInferBase {
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<std::vector<int64_t>> output_shape_;
    bool is_loaded_;

public:
    ONNXRuntimeInference();
    ~ONNXRuntimeInference() override;
    
    bool loadModel(const std::string& model_path);
    cv::Mat convertInput(const cv::Mat& preprocessed_img);
    bool infer(const cv::Mat& input_blob, std::vector<cv::Mat>& outputs) override;
    std::vector<int64_t> getInputShape() const override;
    std::vector<std::vector<int64_t>> getOutputShapes() const override;
    bool isLoaded() const override;
};

/**
 * @brief OpenVINO inference implementation
 */
class OpenVINOInference : public ModelInferBase {
private:
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    std::vector<int64_t> input_shape_;
    std::vector<std::vector<int64_t>> output_shape_;
    bool is_loaded_;

public:
    OpenVINOInference();
    ~OpenVINOInference() override;
    
    bool loadModel(const std::string& model_path, const std::string& device = "CPU", const std::string& cache_dir = "model_compile_cache");
    bool infer(const cv::Mat& input_blob, std::vector<cv::Mat>& outputs) override;
    cv::Mat convertInput(const cv::Mat& preprocessed_img) override;
    std::vector<int64_t> getInputShape() const override;
    std::vector<std::vector<int64_t>> getOutputShapes() const override;
    bool isLoaded() const override;
};
