// C++ port of build_engine.py — builds a TensorRT INT8+FP16 engine from the
// QDQ ONNX graph produced by quantize.py + export_onnx.py (the embedl-deploy
// pipeline — see https://docs.embedl.com/embedl-deploy/latest/auto_tutorials/sam3.html).
// Mirrors the same defaults (constant.py): BUILDER_OPTIMIZATION_LEVEL = 3,
// workspace = 4 GiB, INT8+FP16 hybrid precision.
//
// TensorRT reads the QuantizeLinear/DequantizeLinear nodes in the ONNX graph
// directly (explicit quantization) — no separate INT8 calibrator is needed.
//
// Build (on the GPU box, TensorRT >= 8.6):
//   cmake -B build -DTENSORRT_ROOT=/path/to/TensorRT && cmake --build build
//
// Run:
//   ./build/build_engine [onnx_path] [engine_path] [timing_cache_path] [opt_level]
//   ./build/build_engine \
//       ../artifacts/sam3_resized_924_int8_qdq.onnx \
//       ../artifacts/sam3.engine \
//       ../artifacts/trt_timing.cache \
//       3

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDefaultWorkspaceBytes = 4LL << 30; // 4 GiB
constexpr int32_t kDefaultOptLevel = 3;
constexpr const char* kDefaultOnnxPath = "artifacts/sam3_resized_924_int8_qdq.onnx";
constexpr const char* kDefaultEnginePath = "artifacts/sam3.engine";
constexpr const char* kDefaultTimingCachePath = "artifacts/trt_timing.cache";

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << std::endl;
    }
  }
};

std::vector<char> readFileBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.good()) {
    return {}; // absent is fine — treated as "no prior timing cache"
  }
  const std::streamsize size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(size));
  if (size > 0 && !f.read(buffer.data(), size)) {
    throw std::runtime_error("Failed to read " + path);
  }
  return buffer;
}

void writeFileBytes(const std::string& path, const void* data, size_t size) {
  std::ofstream f(path, std::ios::binary);
  if (!f.good()) {
    throw std::runtime_error("Failed to open " + path + " for writing");
  }
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

} // namespace

int main(int argc, char** argv) {
  const std::string onnxPath = argc > 1 ? argv[1] : kDefaultOnnxPath;
  const std::string enginePath = argc > 2 ? argv[2] : kDefaultEnginePath;
  const std::string timingCachePath = argc > 3 ? argv[3] : kDefaultTimingCachePath;
  const int32_t optLevel = argc > 4 ? std::stoi(argv[4]) : kDefaultOptLevel;

  Logger logger;
  initLibNvInferPlugins(&logger, "");

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    std::cerr << "Failed to create IBuilder" << std::endl;
    return 1;
  }

  const auto explicitBatch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    std::cerr << "Failed to create INetworkDefinition" << std::endl;
    return 1;
  }

  std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, logger));
  if (!parser->parseFromFile(onnxPath.c_str(),
                              static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << parser->getError(i)->desc() << std::endl;
    }
    std::cerr << "Failed to parse " << onnxPath << std::endl;
    return 1;
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  config->setFlag(nvinfer1::BuilderFlag::kINT8);
  config->setBuilderOptimizationLevel(optLevel);
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, kDefaultWorkspaceBytes);

  const std::vector<char> cacheBytes = readFileBytes(timingCachePath);
  std::unique_ptr<nvinfer1::ITimingCache> timingCache(
      config->createTimingCache(cacheBytes.data(), cacheBytes.size()));
  config->setTimingCache(*timingCache, /*ignoreMismatch=*/false);

  std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(
      builder->buildSerializedNetwork(*network, *config));
  if (!serializedEngine) {
    std::cerr << "Engine build failed" << std::endl;
    return 1;
  }

  writeFileBytes(enginePath, serializedEngine->data(), serializedEngine->size());
  std::cout << "  " << enginePath << " ("
            << serializedEngine->size() / 1e9 << " GB)" << std::endl;

  // `timingCache` was updated in place by buildSerializedNetwork(); it's owned
  // by us (created via createTimingCache), not by `config` — do not also fetch
  // and delete config->getTimingCache(), that would double-free the same cache.
  std::unique_ptr<nvinfer1::IHostMemory> serializedCache(timingCache->serialize());
  writeFileBytes(timingCachePath, serializedCache->data(), serializedCache->size());
  std::cout << "  " << timingCachePath << std::endl;

  return 0;
}
