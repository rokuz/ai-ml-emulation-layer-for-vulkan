/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/device.hpp"
#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"

#include <spirv-tools/libspirv.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mlsdk::el::utilities;

namespace {

constexpr const char *usage =
    "usage: mlel_opsuite <manifest> (--golden <dir> | --check <dir>) [--filter <substring>] [--layer <dir>]\n"
    "\n"
    "Runs the graphs of a manifest written by opsuite.py on deterministic inputs. --golden stores the outputs,\n"
    "--check compares them byte by byte with stored ones, --layer loads the emulation layers from a directory.\n";

struct TensorSpec {
    vk::Format format = vk::Format::eUndefined;
    size_t elementSize = 0;
    bool isFloat = false;
    std::vector<int64_t> dimensions;
    uint32_t seed = 0;

    size_t elementCount() const {
        size_t count = 1;
        for (const int64_t dimension : dimensions) {
            count *= static_cast<size_t>(dimension);
        }
        return count;
    }
};

struct GraphSpec {
    std::string name;
    std::filesystem::path path;
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
};

bool parseFormat(const std::string &name, TensorSpec &tensor) {
    const struct {
        const char *name;
        vk::Format format;
        size_t elementSize;
        bool isFloat;
    } formats[] = {
        {"i8", vk::Format::eR8Sint, 1, false},    {"i16", vk::Format::eR16Sint, 2, false},
        {"i32", vk::Format::eR32Sint, 4, false},  {"f16", vk::Format::eR16Sfloat, 2, true},
        {"f32", vk::Format::eR32Sfloat, 4, true},
    };
    for (const auto &format : formats) {
        if (name == format.name) {
            tensor.format = format.format;
            tensor.elementSize = format.elementSize;
            tensor.isFloat = format.isFloat;
            return true;
        }
    }
    return false;
}

std::vector<int64_t> parseDimensions(const std::string &text) {
    std::vector<int64_t> dimensions;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        dimensions.push_back(std::stoll(item));
    }
    return dimensions;
}

std::vector<GraphSpec> parseManifest(const std::filesystem::path &manifest) {
    std::vector<GraphSpec> graphs;
    std::ifstream file(manifest);
    std::string line;
    GraphSpec current;
    while (std::getline(file, line)) {
        std::stringstream stream(line);
        std::string keyword;
        stream >> keyword;
        if (keyword == "graph") {
            std::string path;
            current = GraphSpec();
            stream >> current.name >> path;
            current.path = manifest.parent_path() / path;
        } else if (keyword == "input" || keyword == "output") {
            std::string format;
            std::string dimensions;
            TensorSpec tensor;
            stream >> format >> dimensions;
            if (!parseFormat(format, tensor)) {
                throw std::runtime_error("Unknown tensor format " + format + " in graph " + current.name);
            }
            tensor.dimensions = parseDimensions(dimensions);
            if (keyword == "input") {
                stream >> tensor.seed;
                current.inputs.push_back(tensor);
            } else {
                current.outputs.push_back(tensor);
            }
        } else if (keyword == "end") {
            graphs.push_back(current);
        }
    }
    return graphs;
}

uint32_t xorshift(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

uint16_t toHalf(const double value) {
    if (value == 0.0) {
        return 0;
    }
    const uint32_t sign = value < 0 ? 0x8000U : 0U;
    double magnitude = std::fabs(value);
    int32_t exponent = 0;
    while (magnitude >= 2.0) {
        magnitude /= 2.0;
        exponent++;
    }
    while (magnitude < 1.0) {
        magnitude *= 2.0;
        exponent--;
    }
    const uint32_t mantissa = static_cast<uint32_t>(std::lround((magnitude - 1.0) * 1024.0)) & 0x3ffU;
    return static_cast<uint16_t>(sign | static_cast<uint32_t>((exponent + 15) << 10) | mantissa);
}

std::vector<uint8_t> makeInput(const TensorSpec &tensor) {
    const size_t count = tensor.elementCount();
    std::vector<uint8_t> data(count * tensor.elementSize);
    uint32_t state = tensor.seed * 2654435761U + 0x9e3779b9U;
    if (state == 0) {
        state = 1;
    }
    const int64_t minValue = -(int64_t(1) << (tensor.elementSize * 8 - 1));
    const int64_t maxValue = (int64_t(1) << (tensor.elementSize * 8 - 1)) - 1;
    for (size_t i = 0; i < count; i++) {
        const uint32_t random = xorshift(state);
        const uint32_t kind = random & 15;
        uint8_t *const pointer = data.data() + i * tensor.elementSize;
        if (tensor.isFloat) {
            const double value = kind == 0 ? 0.0 : double(int32_t((random >> 8) % 129) - 64) / 16.0;
            if (tensor.elementSize == 2) {
                const uint16_t half = toHalf(value);
                std::memcpy(pointer, &half, sizeof(half));
            } else {
                const auto single = static_cast<float>(value);
                std::memcpy(pointer, &single, sizeof(single));
            }
            continue;
        }
        int64_t value = 0;
        if (kind == 0) {
            value = minValue;
        } else if (kind == 1) {
            value = maxValue;
        } else if (kind != 2) {
            value = int64_t(int32_t(xorshift(state)));
            value = minValue + (value - int64_t(INT32_MIN)) % (maxValue - minValue + 1);
        }
        std::memcpy(pointer, &value, tensor.elementSize);
    }
    return data;
}

std::vector<uint32_t> assemble(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    const std::string text(std::istreambuf_iterator<char>{file}, {});

    spvtools::SpirvTools tools{SPV_ENV_UNIVERSAL_1_6};
    std::string messages;
    tools.SetMessageConsumer(
        [&messages](spv_message_level_t, const char *, const spv_position_t &position, const char *message) {
            messages += std::to_string(position.line) + ": " + message + "\n";
        });

    std::vector<uint32_t> spirv;
    if (!tools.Assemble(text, &spirv)) {
        throw std::runtime_error("Failed to assemble " + path.string() + "\n" + messages);
    }
    return spirv;
}

void setEnvironment(const char *name, const std::string &value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

std::shared_ptr<Device> createDevice() {
    const std::vector<const char *> layers = {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"};
    const std::vector<const char *> extensions = {
        VK_ARM_DATA_GRAPH_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
        VK_ARM_TENSORS_EXTENSION_NAME,
    };

    vk::PhysicalDeviceFeatures2 features;

    auto context = std::make_shared<vk::raii::Context>();
    auto instance = std::make_shared<Instance>(context, layers);
    auto physicalDevice = std::make_shared<PhysicalDevice>(instance, extensions);

    return std::make_shared<Device>(physicalDevice, extensions, &features);
}

std::vector<uint8_t> dispatch(std::shared_ptr<Device> &device, const GraphSpec &graph) {
    const auto spirv = assemble(graph.path);

    std::vector<std::shared_ptr<Tensor>> outputs;
    GraphPipeline::BindingMap bindings;
    uint32_t binding = 0;
    for (const auto &input : graph.inputs) {
        bindings[binding++] = {
            std::make_shared<Tensor>(device, Shape{input.format, input.dimensions}, makeInput(input))};
    }
    for (const auto &output : graph.outputs) {
        outputs.push_back(std::make_shared<Tensor>(device, Shape{output.format, output.dimensions}));
        outputs.back()->clear();
        bindings[binding++] = {outputs.back()};
    }

    const GraphConstants constants;
    GraphPipeline pipeline(device, {bindings}, constants, spirv);
    pipeline.dispatchSubmit();

    std::vector<uint8_t> result;
    for (size_t i = 0; i < outputs.size(); i++) {
        const size_t size = graph.outputs[i].elementCount() * graph.outputs[i].elementSize;
        result.insert(result.end(), outputs[i]->data(), outputs[i]->data() + size);
    }
    return result;
}

std::vector<uint8_t> readFile(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>{file}, {}};
}

int run(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << usage;
        return 1;
    }

    const std::filesystem::path manifest = argv[1];
    std::filesystem::path goldenDir;
    std::filesystem::path checkDir;
    std::string filter;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--golden") {
            goldenDir = argv[i + 1];
        } else if (option == "--check") {
            checkDir = argv[i + 1];
        } else if (option == "--filter") {
            filter = argv[i + 1];
        } else if (option == "--layer") {
            setEnvironment("VK_LAYER_PATH", argv[i + 1]);
        } else {
            std::cout << usage;
            return 1;
        }
    }
    if (goldenDir.empty() == checkDir.empty()) {
        std::cout << usage;
        return 1;
    }

    const auto graphs = parseManifest(manifest);
    if (graphs.empty()) {
        std::cout << "No graphs in " << manifest.string() << std::endl;
        return 1;
    }
    if (!goldenDir.empty()) {
        std::filesystem::create_directories(goldenDir);
    }

    auto device = createDevice();

    uint32_t passed = 0;
    uint32_t failed = 0;
    uint32_t unsupported = 0;

    for (const auto &graph : graphs) {
        if (!filter.empty() && graph.name.find(filter) == std::string::npos) {
            continue;
        }

        std::vector<uint8_t> record = {1};
        std::string status = "ok";
        try {
            const auto result = dispatch(device, graph);
            record.insert(record.end(), result.begin(), result.end());
        } catch (const std::exception &exception) {
            record = {0};
            status = std::string("unsupported: ") + exception.what();
            unsupported++;
        }

        std::string verdict;
        if (!goldenDir.empty()) {
            std::ofstream file(goldenDir / (graph.name + ".bin"), std::ios::binary);
            file.write(reinterpret_cast<const char *>(record.data()), static_cast<std::streamsize>(record.size()));
            verdict = file.good() ? "stored" : "FAIL cannot write the golden file";
        } else {
            const auto golden = readFile(checkDir / (graph.name + ".bin"));
            if (golden.empty()) {
                verdict = "FAIL no golden";
            } else if (golden == record) {
                verdict = "ok";
            } else {
                size_t differing = 0;
                for (size_t i = 0; i < std::min(golden.size(), record.size()); i++) {
                    differing += golden[i] != record[i] ? 1 : 0;
                }
                verdict =
                    "FAIL " + std::to_string(differing) + " of " + std::to_string(record.size()) + " bytes differ";
            }
        }
        if (verdict.rfind("FAIL", 0) == 0) {
            failed++;
        } else {
            passed++;
        }

        std::cout << std::left << std::setw(56) << graph.name << " " << std::setw(12) << verdict << " " << status
                  << std::endl;
    }

    std::cout << (failed != 0 ? "FAILED" : "PASSED") << ": " << passed << " passed, " << failed << " failed, "
              << unsupported << " unsupported" << std::endl;
    return failed != 0 ? 1 : 0;
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception &exception) {
        std::cout << "ERROR: " << exception.what() << std::endl;
        return 2;
    }
}
