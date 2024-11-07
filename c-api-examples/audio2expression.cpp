#include "onnxruntime_cxx_api.h"
#include "../sherpa-onnx/c-api/c-api.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>

#define SHERPA_ONNX_LOGE(...)                                            \
  do {                                                                   \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__,                     \
            static_cast<int>(__LINE__));                                 \
    fprintf(stderr, ##__VA_ARGS__);                                      \
    fprintf(stderr, "\n");                                               \
  } while (0)


int main() {
    std::string audio_path = "E:\\github\\sherpa-onnx\\build\\aaa.wav";
    SHERPA_ONNX_LOGE("audio_path=%s", audio_path.c_str());

    auto start_time = std::chrono::high_resolution_clock::now();

    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    SHERPA_ONNX_LOGE("session created 1");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    SHERPA_ONNX_LOGE("session created 2");
    Ort::Session session(env, L"E:\\github\\sherpa-onnx\\build\\A2E_LSTM_10311254_HS156_NL4.onnx", session_options);
    SHERPA_ONNX_LOGE("session created");

    SherpaOnnxFeatureConfig config;
    config.sample_rate = 16000;
    config.feature_dim = 80;
    config.normalize_samples = 1;
    config.is_mfcc = 1;
    config.num_ceps = 272;
    config.frame_shift_ms = 33.33333333333333;
    config.frame_length_ms = 33.33333333333333;
    SherpaOnnxFeatureExtractor *extractor = SherpaOnnxCreateFeatureExtractor(&config);

    // Load audio file
    std::vector<float> samples;
    int32_t sample_rate = 16000;
    int32_t n = 0;
    const SherpaOnnxWave* wave = SherpaOnnxReadWave(audio_path.c_str());
    if (wave == nullptr) {
        SHERPA_ONNX_LOGE("Failed to read wave file");
        return -1;
    }
    samples.assign(wave->samples, wave->samples + wave->num_samples);
    n = wave->num_samples;
    SherpaOnnxFreeWave(wave);
    SHERPA_ONNX_LOGE("sample_rate=%d, n=%d", sample_rate, n);
    SherpaOnnxFeatureExtractorAcceptWaveform(extractor, 16000, samples.data(), n);
    SherpaOnnxFeature feature = SherpaOnnxFeatureExtractorGetFeature(extractor);
    int64_t time_steps = feature.data_size / feature.feature_dim;
    SHERPA_ONNX_LOGE("data_size: %d, feature_dim: %d, time_steps: %d", feature.data_size, feature.feature_dim, time_steps);
    int64_t feature_size = feature.feature_dim;

    // 确定的输入和输出名称
    const char* input_name = "input";
    const char* length_name = "length";
    const char* mask_name = "mask";
    const char* output_name = "output";

    std::vector<float> input_tensor_values(1 * time_steps * feature_size, 0.0f); // 根据实际输入形状初始化
    std::vector<int64_t> input_tensor_shape = {1, time_steps, feature_size};
    std::vector<int64_t> length_tensor_values = {time_steps};
    std::vector<int64_t> length_tensor_shape = {1};

    std::vector<float> mask_tensor_values(1 * time_steps, 1.0f); // 全1的mask
    std::vector<int64_t> mask_tensor_shape = {1, time_steps};
    // 打印输入数据的长度和形状
    std::cout << "Input tensor shape: [";
    for (size_t i = 0; i < input_tensor_shape.size(); ++i) {
        std::cout << input_tensor_shape[i];
        if (i < input_tensor_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Input tensor values size: " << input_tensor_values.size() << std::endl;

    std::cout << "Length tensor shape: [";
    for (size_t i = 0; i < length_tensor_shape.size(); ++i) {
        std::cout << length_tensor_shape[i];
        if (i < length_tensor_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Length tensor values size: " << length_tensor_values.size() << std::endl;

    std::cout << "Mask tensor shape: [";
    for (size_t i = 0; i < mask_tensor_shape.size(); ++i) {
        std::cout << mask_tensor_shape[i];
        if (i < mask_tensor_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Mask tensor values size: " << mask_tensor_values.size() << std::endl;

     // 创建 ONNX Runtime 张量
    Ort::AllocatorWithDefaultOptions allocator;
    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());
    SHERPA_ONNX_LOGE("input_tensor created, time_steps=%d, feature_size=%d", time_steps, feature_size);
    const Ort::MemoryInfo memory_info2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(memory_info2, length_tensor_values.data(), length_tensor_values.size(), length_tensor_shape.data(), length_tensor_shape.size());
    SHERPA_ONNX_LOGE("length_tensor created, length_data size=%d", length_tensor_values.size());
    const Ort::MemoryInfo memory_info3 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(memory_info3, mask_tensor_values.data(), mask_tensor_values.size(), mask_tensor_shape.data(), mask_tensor_shape.size());
    SHERPA_ONNX_LOGE("mask_tensor created, mask_data size=%d", mask_tensor_values.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(length_tensor));
    input_tensors.push_back(std::move(mask_tensor));

    // 运行推理
    const char* input_names[] = {input_name, length_name, mask_name};
    const char* output_names[] = {output_name};
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 3, output_names, 1);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    // 打印output_tensors的形状
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        std::vector<int64_t> output_tensor_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output tensor shape: [";
        for (size_t j = 0; j < output_tensor_shape.size(); ++j) {
            std::cout << output_tensor_shape[j];
            if (j < output_tensor_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    // 上面打印为：Output tensor shape: [1, 230, 272]
    // 说明输出的形状为[1, 230, 272]
    // 现将output_tensors存入csv文件，即供230行，每行272个数据
    std::ofstream output_file("output.csv");
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        std::vector<int64_t> output_tensor_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        const float* output_tensor_values = output_tensors[i].GetTensorMutableData<float>();
        for (int64_t j = 0; j < output_tensor_shape[1]; ++j) {
            for (int64_t k = 0; k < output_tensor_shape[2]; ++k) {
                output_file << output_tensor_values[j * output_tensor_shape[2] + k];
                if (k < output_tensor_shape[2] - 1) output_file << ",";
            }
            output_file << std::endl;
        }
    }
    output_file.close();

    return 0;
}
