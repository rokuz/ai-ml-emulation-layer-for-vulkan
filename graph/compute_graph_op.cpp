/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute_graph_op.hpp"
#include "graph_log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string_view>

using namespace mlsdk::el::log;
using namespace mlsdk::el::utils;

namespace mlsdk::el::compute {

namespace {
void makeAndConnectVirtualTensor(const std::shared_ptr<TensorDescriptor> &tensor,
                                 graph_op::ComputePipelineBase *descendant) {
    auto *parent = tensor->getPipeline();
    auto virtualTensor = std::make_shared<VirtualTensor>(tensor, descendant);

    if (parent != nullptr) {
        parent->pushDescendant(virtualTensor);
    }

    if (descendant != nullptr) {
        descendant->pushParent(virtualTensor);
    }
}
} // namespace

VkFormat accTypeVkFormat(uint32_t accType) {
    switch (accType) {
    case 1:
        return VK_FORMAT_R32_SINT;
    case 2:
        return VK_FORMAT_R16_SFLOAT;
    case 3:
        return VK_FORMAT_R32_SFLOAT;
    case 4:
        return VK_FORMAT_R64_SINT;
    default:
        throw std::runtime_error("Unsupported acc type " + std::to_string(accType));
    }
}

namespace {

std::string_view accTypeString(uint32_t accType) {
    switch (accType) {
    case 1:
        return "int32_t";
    case 2:
        return "float16_t";
    case 3:
        return "float";
    case 4:
        return "int64_t";
    default:
        throw std::runtime_error("Unsupported AVG_POOL2D acc type " + std::to_string(accType));
    }
}

} // namespace

namespace graph_op {

/*******************************************************************************
 * ComputeDescriptorSet
 *******************************************************************************/

ComputeDescriptorSet::ComputeDescriptorSet(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, const VkDevice _device,
    VkDescriptorPool _descriptorPool, VkDescriptorSet _descriptorSet,
    const std::vector<DescriptorSetTensorBinding> &_tensorBindings)
    : loader{_loader}, device{_device}, descriptorPool{_descriptorPool}, descriptorSet{_descriptorSet} {
    for (const auto &descriptorBinding : _tensorBindings) {
        const auto key = std::make_tuple(descriptorBinding.binding, descriptorBinding.arrayIndex);
        tensorMap[key] = descriptorBinding.tensor;
        tensorHandleMap[key] = descriptorBinding.tensor->getVkTensorARM();
        tensorViewMap[key] = descriptorBinding.tensor->getVkTensorViewARM();
        tensorDescriptorMap[descriptorBinding.tensorDescriptor].push_back(key);
    }
}

ComputeDescriptorSet::~ComputeDescriptorSet() {
    loader->vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    loader->vkDestroyDescriptorPool(device, descriptorPool, nullptr);
}

VkDescriptorSet ComputeDescriptorSet::getVkDescriptorSet() const { return descriptorSet; }

bool ComputeDescriptorSet::KeyCompare::operator()(const TensorBindingKey &a, const TensorBindingKey &b) const {
    return a < b;
}

std::vector<std::shared_ptr<Tensor>> ComputeDescriptorSet::getTensors() const {
    std::vector<std::shared_ptr<Tensor>> tensors;
    tensors.reserve(tensorMap.size());
    for (const auto &entry : tensorMap) {
        tensors.push_back(entry.second);
    }
    return tensors;
}

VkTensorARM ComputeDescriptorSet::getVkTensorARM(const uint32_t binding, const uint32_t arrayIndex) const {
    return tensorHandleMap.at(std::make_tuple(binding, arrayIndex));
}

bool ComputeDescriptorSet::updateDescriptorSet(const std::shared_ptr<TensorDescriptor> &tensorDescriptor,
                                               const VkTensorARM tensor, const VkTensorViewARM tensorView) {
    if (const auto tensorDescriptorIt = tensorDescriptorMap.find(tensorDescriptor);
        tensorDescriptorIt != tensorDescriptorMap.end()) {
        for (const auto &[binding, arrayIndex] : tensorDescriptorIt->second) {
            tensorHandleMap[{binding, arrayIndex}] = tensor;
            tensorViewMap[{binding, arrayIndex}] = tensorView;
            updateDescriptorSet(binding, arrayIndex);
        }
        return true;
    }

    return false;
}

void ComputeDescriptorSet::updateDescriptorSet() {
    for (const auto &[key, _] : tensorMap) {
        updateDescriptorSet(std::get<0>(key), std::get<1>(key));
    }
}

void ComputeDescriptorSet::updateDescriptorSet(const uint32_t binding, const uint32_t arrayIndex) {
    const auto key = std::make_tuple(binding, arrayIndex);
    auto *const tensorView = tensorViewMap.at(key);

    const VkWriteDescriptorSetTensorARM descriptorInfo = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM, // type
        nullptr,                                           // next
        1,                                                 // tensor view count
        &tensorView,                                       // tensor views
    };

    const VkWriteDescriptorSet writeDescriptorSet = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // type
        &descriptorInfo,                        // next
        descriptorSet,                          // descriptor set
        binding,                                // binding
        arrayIndex,                             // dst array element
        1,                                      // descriptor count
        VK_DESCRIPTOR_TYPE_TENSOR_ARM,          // descriptor type
        nullptr,                                // image info
        nullptr,                                // buffer info
        nullptr,                                // texel buffer view
    };

    loader->vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
}

/*******************************************************************************
 * ComputePipelineLayout
 *******************************************************************************/

ComputePipelineLayout::ComputePipelineLayout(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
    DescriptorMap _descriptorMap, const PushConstant &_pushConstant)
    : loader{_loader}, device{_device}, descriptorMap{std::move(_descriptorMap)}, pushConstant{_pushConstant} {
    std::set<uint32_t> usedSets;
    for (auto &descriptor : descriptorMap) {
        if (descriptor.id.set != UINT32_MAX) {
            usedSets.insert(descriptor.id.set);
        }
    }

    uint32_t nextSet = 0;
    for (auto &descriptor : descriptorMap) {
        if (descriptor.id.set == UINT32_MAX) {
            while (usedSets.count(nextSet) > 0) {
                nextSet++;
            }
            descriptor.id.set = nextSet;
            usedSets.insert(nextSet);
        }
    }

    descriptorSetLayouts = createDescriptorSetLayouts();
    pipelineLayout = createPipelineLayout();
}

ComputePipelineLayout::~ComputePipelineLayout() {
    for (auto *descriptorSetLayout : descriptorSetLayouts) {
        loader->vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    loader->vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

VkPipelineLayout ComputePipelineLayout::getVkPipelineLayout() const { return pipelineLayout; }

const DescriptorMap &ComputePipelineLayout::getDescriptorMap() const { return descriptorMap; }

const PushConstant &ComputePipelineLayout::getPushConstant() const { return pushConstant; }

const std::shared_ptr<TensorDescriptor> &ComputePipelineLayout::getTensorForSet(const uint32_t set) const {
    for (const auto &descriptor : descriptorMap) {
        if (descriptor.id.set == set && descriptor.direction == Output) {
            return descriptor.tensor;
        }
    }

    for (const auto &descriptor : descriptorMap) {
        if (descriptor.id.set == set) {
            return descriptor.tensor;
        }
    }

    throw std::runtime_error("Tensor descriptor not found for set " + std::to_string(set));
}

void ComputePipelineLayout::cmdBindAndDispatch(VkCommandBuffer commandBuffer,
                                               const ComputeDescriptorSetMap &descriptorSetMap) {
    cmdBindDescriptorSets(commandBuffer, descriptorSetMap);
    cmdPushConstants(commandBuffer);
    cmdPipelineBarrier(commandBuffer, descriptorSetMap);
}

void ComputePipelineLayout::cmdBindDescriptorSets(VkCommandBuffer commandBuffer,
                                                  const ComputeDescriptorSetMap &descriptorSetMap) {
    std::set<uint32_t> boundSets;
    for (const auto &descriptor : descriptorMap) {
        boundSets.insert(descriptor.id.set);
    }

    for (const auto set : boundSets) {
        const auto &descriptorSet = descriptorSetMap.at({pipelineLayout, set});

        auto *const vkDescriptorSet = descriptorSet->getVkDescriptorSet();
        loader->vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, set, 1,
                                        &vkDescriptorSet, 0, nullptr);
    }
}

void ComputePipelineLayout::cmdPushConstants(VkCommandBuffer commandBuffer) {
    if (pushConstant.pointer != nullptr) {
        loader->vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstant.size,
                                   pushConstant.pointer);
    }
}

void ComputePipelineLayout::cmdPipelineBarrier(VkCommandBuffer commandBuffer,
                                               const ComputeDescriptorSetMap &descriptorSetMap) const {
    std::vector<VkTensorMemoryBarrierARM> tensorMemoryBarriers;

    for (const auto &descriptor : descriptorMap) {
        // Only add barriers for input tensors
        if (descriptor.direction != Input) {
            continue;
        }

        const auto set = descriptor.id.set;
        const auto binding = descriptor.id.binding;
        const auto arrayIndex = descriptor.id.arrayIndex;
        const auto &descriptorSet = descriptorSetMap.at({pipelineLayout, set});

        tensorMemoryBarriers.push_back({
            VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_ARM,        // type
            nullptr,                                            // next
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,             // src stage mask
            VK_ACCESS_2_SHADER_WRITE_BIT,                       // src access mask
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,             // dst stage mask
            VK_ACCESS_2_SHADER_READ_BIT,                        // dst access mask
            VK_QUEUE_FAMILY_IGNORED,                            // src queue family index
            VK_QUEUE_FAMILY_IGNORED,                            // dst queue family index
            descriptorSet->getVkTensorARM(binding, arrayIndex), // tensor
        });
    }

    const VkTensorDependencyInfoARM tensorDependencyInfo = {
        VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM,       // type
        nullptr,                                            // next
        static_cast<uint32_t>(tensorMemoryBarriers.size()), // tensorMemoryBarrierCount
        tensorMemoryBarriers.data()                         // pTensorMemoryBarriers
    };

    const VkDependencyInfo dependencyInfo = {
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO, // type
        &tensorDependencyInfo,             // next
        0,                                 // dependencyFlags
        0,                                 // memoryBarrierCount
        nullptr,                           // pMemoryBarriers
        0,                                 // bufferMemoryBarrierCount
        nullptr,                           // pBufferMemoryBarriers
        0,                                 // imageMemoryBarrierCount
        nullptr                            // pImageMemoryBarriers
    };

    loader->vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

std::vector<VkDescriptorSetLayoutBinding>
ComputePipelineLayout::getDescriptorSetLayoutBinding(const uint32_t set) const {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;

    std::map<uint32_t, uint32_t> bindingCountMap;
    for (const auto &descriptor : descriptorMap) {
        if (descriptor.id.set != set) {
            continue;
        }

        auto &count = bindingCountMap[descriptor.id.binding];
        count = std::max(count, descriptor.id.arrayIndex + 1);
    }

    for (const auto &[binding, descriptorCount] : bindingCountMap) {
        const VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {
            binding,                       // binding
            VK_DESCRIPTOR_TYPE_TENSOR_ARM, // descriptor type
            descriptorCount,               // descriptor count
            VK_SHADER_STAGE_COMPUTE_BIT,   // type
            nullptr,                       // sampler
        };
        descriptorSetLayoutBindings.emplace_back(descriptorSetLayoutBinding);
    }

    return descriptorSetLayoutBindings;
}

std::vector<VkDescriptorSetLayout> ComputePipelineLayout::createDescriptorSetLayouts() const {
    if (descriptorMap.empty()) {
        return {};
    }

    uint32_t maxSet = 0;
    for (const auto &descriptor : descriptorMap) {
        maxSet = std::max(maxSet, descriptor.id.set);
    }

    std::vector<VkDescriptorSetLayout> layouts(maxSet + 1, nullptr);
    for (uint32_t set = 0; set <= maxSet; set++) {
        const auto &descriptorSetLayoutBindings = getDescriptorSetLayoutBinding(set);

        std::vector<VkDescriptorSetLayoutCreateFlags> bindingFlags(descriptorSetLayoutBindings.size(),
                                                                   VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);

        const VkDescriptorSetLayoutBindingFlagsCreateInfo descriptorSetBindingFlagsCreateInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,   // type
            nullptr,                                                             // next
            static_cast<uint32_t>(descriptorSetLayoutBindings.size()),           // binding count
            descriptorSetLayoutBindings.empty() ? nullptr : bindingFlags.data(), // binding flags
        };

        const VkDescriptorSetLayoutCreateInfo descriptorSetCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,                                  // type
            descriptorSetLayoutBindings.empty() ? nullptr : &descriptorSetBindingFlagsCreateInfo, // next
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,                           // flags
            static_cast<uint32_t>(descriptorSetLayoutBindings.size()),                            // binding count
            descriptorSetLayoutBindings.empty() ? nullptr : descriptorSetLayoutBindings.data(),   // bindings
        };

        if (loader->vkCreateDescriptorSetLayout(device, &descriptorSetCreateInfo, nullptr, &layouts[set]) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }
    }

    return layouts;
}

VkDescriptorPool ComputePipelineLayout::createDescriptorPool(const uint32_t set) const {
    uint32_t descriptorCount = 0;
    for (const auto &binding : getDescriptorSetLayoutBinding(set)) {
        descriptorCount += binding.descriptorCount;
    }

    const VkDescriptorPoolSize descriptorPoolSize = {
        VK_DESCRIPTOR_TYPE_TENSOR_ARM, // type
        descriptorCount,               // descriptor count
    };

    const VkDescriptorPoolCreateFlags descriptorPoolCreateFlags =
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

    const VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, // type
        nullptr,                                       // next
        descriptorPoolCreateFlags,                     // flags,
        1,                                             // max sets
        1,                                             // pool size count
        &descriptorPoolSize,                           // descriptor pool size
    };

    VkDescriptorPool pool;
    if (loader->vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &pool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor pool");
    }

    return pool;
}

void ComputePipelineLayout::makeDescriptorSets(ComputeDescriptorSetMap &mapping,
                                               const TensorDescriptorMap &filter) const {
    std::map<uint32_t, std::vector<DescriptorSetTensorBinding>> bindingsPerSet;
    std::set<uint32_t> allSets;

    for (const auto &descriptor : descriptorMap) {
        allSets.insert(descriptor.id.set);

        if (const auto filterIt = filter.find(descriptor.tensor); filterIt != filter.end()) {
            bindingsPerSet[descriptor.id.set].push_back({
                descriptor.id.binding,
                descriptor.id.arrayIndex,
                descriptor.tensor,
                filterIt->second,
            });
        }
    }

    for (const auto set : allSets) {
        const auto bindingIt = bindingsPerSet.find(set);
        if (bindingIt == bindingsPerSet.end()) {
            continue;
        }

        const auto expectedBindings =
            static_cast<size_t>(std::count_if(descriptorMap.begin(), descriptorMap.end(),
                                              [set](const auto &descriptor) { return descriptor.id.set == set; }));
        if (bindingIt->second.size() != expectedBindings) {
            throw std::runtime_error("Descriptor set " + std::to_string(set) +
                                     " has split ownership across descriptor sources");
        }

        auto *vkDescriptorPool = createDescriptorPool(set);
        mapping[{pipelineLayout, set}] = std::make_shared<ComputeDescriptorSet>(
            loader, device, vkDescriptorPool, createDescriptorSet(vkDescriptorPool, set), bindingIt->second);
    }
}

VkDescriptorSet ComputePipelineLayout::createDescriptorSet(const VkDescriptorPool vkDescriptorPool,
                                                           const uint32_t set) const {
    // Allocate descriptor set
    const VkDescriptorSetAllocateInfo descriptorSetAllocInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // type
        nullptr,                                        // next
        vkDescriptorPool,                               // descriptor pool
        1,                                              // descriptor set count
        &descriptorSetLayouts.at(set),                  // descriptor layout set
    };

    VkDescriptorSet descriptorSet;
    if (loader->vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    return descriptorSet;
}

VkPipelineLayout ComputePipelineLayout::createPipelineLayout() const {
    const VkPushConstantRange pushConstantRange = {
        VK_SHADER_STAGE_COMPUTE_BIT, // flags
        0,                           // offset
        pushConstant.size,           // size
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,      // type
        nullptr,                                            // next
        0,                                                  // flags
        static_cast<uint32_t>(descriptorSetLayouts.size()), // layout count
        descriptorSetLayouts.data(),                        // layout
        pushConstant.pointer != nullptr ? 1u : 0u,          // push constant count
        &pushConstantRange,                                 // push constants
    };

    VkPipelineLayout layout;
    if (loader->vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    return layout;
}

/*******************************************************************************
 * ComputePipelineBase
 *******************************************************************************/

ComputePipelineBase::ComputePipelineBase(const std::shared_ptr<ComputePipelineLayout> &_pipelineLayout,
                                         std::string _debugName)
    : pipelineLayout{_pipelineLayout}, debugName{std::move(_debugName)} {}

ComputePipelineBase::~ComputePipelineBase() = default;

void ComputePipelineBase::cmdBindAndDispatch(VkCommandBuffer, const ComputeDescriptorSetMap &) {}

const std::shared_ptr<ComputePipelineLayout> &ComputePipelineBase::getComputePipelineLayout() const {
    return pipelineLayout;
}

const std::vector<std::shared_ptr<VirtualTensor>> &ComputePipelineBase::getParents() const { return parents; }

void ComputePipelineBase::pushParent(const std::shared_ptr<VirtualTensor> &tensor) { parents.emplace_back(tensor); }

const std::vector<std::shared_ptr<VirtualTensor>> &ComputePipelineBase::getDescendants() const { return descendants; }

void ComputePipelineBase::pushDescendant(const std::shared_ptr<VirtualTensor> &tensor) {
    descendants.push_back(tensor);
}

const std::string &ComputePipelineBase::getDebugName() const { return debugName; }

/*******************************************************************************
 * ComputePipeline
 *******************************************************************************/

namespace {
std::shared_ptr<ComputePipelineLayout>
createPipelineLayout(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                     VkDevice device, DescriptorMap descriptorMap, const PushConstant &pushConstant) {
    auto pipelineLayout =
        std::make_shared<ComputePipelineLayout>(loader, device, std::move(descriptorMap), pushConstant);

    return pipelineLayout;
}
} // namespace

ComputePipeline::ComputePipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                 VkDevice _device, DescriptorMap descriptorMap, const PushConstant &pushConstant,
                                 const std::shared_ptr<PipelineCache> &_pipelineCache, const SpirvBinary &_spirv,
                                 const std::string &debugName, const SpecConstants &_constants,
                                 const uint32_t _dispatchSet)
    : ComputePipelineBase(createPipelineLayout(_loader, _device, std::move(descriptorMap), pushConstant), debugName),
      loader{_loader}, device{_device}, pipelineCache{_pipelineCache}, dispatchSet{_dispatchSet},
      // Vulkan objects created from the provided SPIR-V.
      shaderModule{createShaderModule(_spirv)}, pipeline{VK_NULL_HANDLE}, constants{_constants} {
    assert(std::to_string(warp1D) == warp1DSv);
    connectPipelines();
}

void ComputePipeline::finalize() {
    if (pipeline != VK_NULL_HANDLE) {
        return;
    }
    pipeline = createComputePipeline(constants);
    setDebugUtilsObjectName(loader, device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(pipeline), debugName);
}

ComputePipeline::~ComputePipeline() {
    loader->vkDestroyPipeline(device, pipeline, nullptr);
    loader->vkDestroyShaderModule(device, shaderModule, nullptr);
}

void ComputePipeline::cmdBindAndDispatch(VkCommandBuffer commandBuffer,
                                         const ComputeDescriptorSetMap &descriptorSetMap) {
    assert(pipeline != VK_NULL_HANDLE);
    loader->vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    pipelineLayout->cmdBindAndDispatch(commandBuffer, descriptorSetMap);
    cmdDispatch(commandBuffer);
}

void ComputePipeline::cmdDispatch(VkCommandBuffer commandBuffer) {
    // Get first output tensor
    const auto &tensor = pipelineLayout->getTensorForSet(0);
    const auto &size = uint32_t(tensor->getShapeSize());

    const auto groupCountX = static_cast<uint32_t>(std::ceil(std::sqrt(double(divideRoundUp(size, warp1D)))));
    const auto groupCountY = groupCountX;

    loader->vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
}

void ComputePipeline::cmdDispatchVector(VkCommandBuffer commandBuffer, const uint32_t set,
                                        const uint32_t valuesPerInvocation) {
    const auto &dimensions = pipelineLayout->getTensorForSet(set)->getDimensions();
    uint32_t size = divideRoundUp(static_cast<uint32_t>(dimensions.back()), valuesPerInvocation);
    for (size_t i = 0; i + 1 < dimensions.size(); i++) {
        size *= static_cast<uint32_t>(dimensions[i]);
    }

    const auto groupCount = static_cast<uint32_t>(std::ceil(std::sqrt(double(divideRoundUp(size, warp1D)))));
    loader->vkCmdDispatch(commandBuffer, groupCount, groupCount, 1);
}

VkPipeline ComputePipeline::createComputePipeline(const SpecConstants &_constants) const {
    static const bool specialize = []() {
        const char *const value = std::getenv("VMEL_DISABLE_SPECIALIZATION");
        return value == nullptr || std::string_view(value) == "0";
    }();
    constexpr uint32_t dispatchRankId = 100;
    constexpr uint32_t maxDispatchRank = 6;
    std::vector<uint32_t> values = _constants;
    std::vector<uint32_t> ids(_constants.size());
    for (size_t i = 0; i < ids.size(); i++) {
        ids[i] = static_cast<uint32_t>(i);
    }
    const auto &tensor = pipelineLayout->getTensorForSet(dispatchSet);
    if (specialize && tensor != nullptr && tensor->getRank() <= maxDispatchRank) {
        const auto &dimensions = tensor->getDimensions();
        values.push_back(static_cast<uint32_t>(tensor->getRank()));
        ids.push_back(dispatchRankId);
        for (uint32_t i = 0; i < maxDispatchRank; i++) {
            values.push_back(i < dimensions.size() ? static_cast<uint32_t>(dimensions[i]) : 1);
            ids.push_back(dispatchRankId + 1 + i);
        }
    }

    constexpr uint32_t pushWordsValidId = 199;
    constexpr uint32_t pushWordBaseId = 200;
    constexpr uint32_t maxPushWords = 16;
    const auto &push = pipelineLayout->getPushConstant();
    if (specialize && push.pointer != nullptr && push.size >= sizeof(uint32_t)) {
        values.push_back(1);
        ids.push_back(pushWordsValidId);
        const uint32_t words = std::min(push.size / static_cast<uint32_t>(sizeof(uint32_t)), maxPushWords);
        for (uint32_t i = 0; i < words; i++) {
            uint32_t word = 0;
            std::memcpy(&word, static_cast<const char *>(push.pointer) + i * sizeof(uint32_t), sizeof(word));
            values.push_back(word);
            ids.push_back(pushWordBaseId + i);
        }
    }

    const common::SpecializationConstantsView specializationConstants = {
        values.data(),
        static_cast<uint32_t>(values.size() * sizeof(uint32_t)),
        static_cast<uint32_t>(values.size()),
        ids.data(),
    };
    const auto *specialization = values.empty() ? nullptr : &specializationConstants;

    return common::createComputePipeline(loader, device, pipelineCache->getPipelineCache(), shaderModule,
                                         pipelineLayout->getVkPipelineLayout(), specialization);
}

VkShaderModule ComputePipeline::createShaderModule(const SpirvBinary &code) const {
    return common::createShaderModule(loader, device, code);
}

void ComputePipeline::connectPipelines() {
    const auto &descriptorMap = pipelineLayout->getDescriptorMap();

    // Set current pipeline as producer for output tensors
    for (const auto &descriptor : descriptorMap) {
        if (descriptor.direction == Output) {
            descriptor.tensor->setPipeline(this);
        }
    }

    // Create connections to parent pipelines
    for (const auto &descriptor : descriptorMap) {
        if (descriptor.direction == Input) {
            makeAndConnectVirtualTensor(descriptor.tensor, this);
        }
    }
}

/*******************************************************************************
 * Argmax
 *******************************************************************************/

Argmax::Argmax(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _nanMode,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _input), debugName,
                      {_input->getRank(), _output->getRank()}),
      pushConstant{createPushConstant(_axis, _nanMode)} {}

Argmax::PushConstant Argmax::createPushConstant(const uint32_t axis, const uint32_t nanMode) const {
    PushConstant constant = {
        axis,
        nanMode,
    };

    return constant;
}

DescriptorMap Argmax::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Argmax::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input) const {
    const auto *inType = getFormatInfo(input->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_t_type%", inType->typeId},
                                      {"%in_t_lowest%", inType->lowest},
                                      {"%in_t%", inType->glslType},
                                      {"%in_t_comp%", inType->compType},
                                  });
}

/*******************************************************************************
 * ArithmeticRightShift
 *******************************************************************************/

ArithmeticRightShift::ArithmeticRightShift(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
    const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
    const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
    const bool _round, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _output),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache, createSpirv(_pipelineCache, _output),
                      debugName, {_input1->getRank(), _output->getRank()}),
      pushConstant{createPushConstant(_round)} {}

ArithmeticRightShift::PushConstant ArithmeticRightShift::createPushConstant(const bool round) const {
    PushConstant constant = {
        round,
    };
    return constant;
}

DescriptorMap ArithmeticRightShift::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                                        const std::shared_ptr<TensorDescriptor> &input2,
                                                        const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
    };

    return descriptorMap;
}

SpirvBinary ArithmeticRightShift::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                              const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

/*******************************************************************************
 * AvgPool2D
 *******************************************************************************/

AvgPool2D::AvgPool2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                     const std::shared_ptr<TensorDescriptor> &_input, const std::shared_ptr<TensorDescriptor> &_output,
                     const std::vector<int32_t> &_kernel, const std::vector<int32_t> &_stride,
                     const std::vector<int32_t> &_pad, const uint32_t _accType, const int8_t _inputZeroPoint,
                     const int8_t _outputZeroPoint, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output, _accType), debugName),
      pushConstant{createPushConstant(_kernel, _stride, _pad, _inputZeroPoint, _outputZeroPoint)} {}

AvgPool2D::PushConstant AvgPool2D::createPushConstant(const std::vector<int32_t> &kernel,
                                                      const std::vector<int32_t> &stride,
                                                      const std::vector<int32_t> &pad, const int8_t inputZeroPoint,
                                                      const int8_t outputZeroPoint) const {
    PushConstant constant = {
        {
            kernel[0],
            kernel[1],
        },
        {
            stride[0],
            stride[1],
        },
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
        },
        inputZeroPoint,
        outputZeroPoint,
    };

    return constant;
}

DescriptorMap AvgPool2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                             const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary AvgPool2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                   const std::shared_ptr<TensorDescriptor> &output, const uint32_t accType) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    const auto accTypeStr = accTypeString(accType);
    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                      accTypeStr,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%acc_t%", accTypeStr},
                                      {"%in_out_t_lowest%", inOutType->lowest},
                                      {"%in_out_t_max%", inOutType->max},
                                      {"%in_out_t%", inOutType->glslType},
                                      {"%in_out_t_type%", inOutType->typeId},
                                      {"%in_out_t_comp%", inOutType->compType},
                                  });
}

void AvgPool2D::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Cast
 *******************************************************************************/

Cast::Cast(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output), debugName, {_input->getRank()}) {}

DescriptorMap Cast::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                        const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Cast::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                              const std::shared_ptr<TensorDescriptor> &input,
                              const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inType = getFormatInfo(input->getFormat());
    const auto *outType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType,
                                      outType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_t_type%", inType->typeId},
                                      {"%out_t_type%", outType->typeId},
                                      {"%in_t_lowest%", inType->lowest},
                                      {"%in_t_max%", inType->max},
                                      {"%out_t_lowest%", outType->lowest},
                                      {"%out_t_max%", outType->max},
                                      {"%in_t%", inType->glslType},
                                      {"%out_t%", outType->glslType},
                                  });
}

void Cast::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Clamp
 *******************************************************************************/

Clamp::Clamp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
             const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
             const std::shared_ptr<TensorDescriptor> &_output, const real_t _min, const real_t _max,
             const uint32_t _nanMode, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output), debugName, {_input->getRank()}),
      pushConstant{createPushConstant(_min, _max, _nanMode)} {}

Clamp::PushConstant Clamp::createPushConstant(const real_t min, const real_t max, const uint32_t nanMode) const {
    PushConstant constant = {
        min,
        max,
        nanMode,
    };

    return constant;
}

DescriptorMap Clamp::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output) const {
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Clamp::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                               const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                      {"%in_out_t_type%", inOutType->typeId},
                                      {"%in_out_t_comp%", inOutType->compType},
                                  });
}

void Clamp::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Concat
 *******************************************************************************/

Concat::Concat(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _offset,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output), debugName, {_input->getRank()}, 1),
      pushConstant{createPushConstant(_axis, _offset)} {}

Concat::PushConstant Concat::createPushConstant(const uint32_t axis, const uint32_t offset) const {
    PushConstant constant = {
        axis,
        offset,
    };

    return constant;
}

DescriptorMap Concat::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

void Concat::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 1, 4); }

SpirvBinary Concat::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

namespace {

SpecConstants rescaleTailConstants(const RescaleTail *tail) {
    if (tail == nullptr) {
        return {};
    }
    return {tail->scale32, tail->doubleRound, tail->perChannel, static_cast<uint32_t>(tail->inputZeroPoint),
            static_cast<uint32_t>(tail->outputZeroPoint)};
}

void appendRescaleTail(DescriptorMap &descriptorMap, const RescaleTail *tail) {
    if (tail != nullptr) {
        descriptorMap.emplace_back(Input, tail->multiplier);
        descriptorMap.emplace_back(Input, tail->shift);
    }
}

void appendRescaleTailReplacements(PipelineCache::KeyList &keys, PipelineCache::ReplaceList &replacements,
                                   const std::shared_ptr<TensorDescriptor> &output, const RescaleTail *tail) {
    replacements.emplace_back("%rescale_tail%", tail != nullptr ? "1" : "0");
    if (tail == nullptr) {
        return;
    }
    const auto *tailOutType = getFormatInfo(output->getFormat());
    const auto *tailMulType = getFormatInfo(tail->multiplier->getFormat());
    keys.push_back(tailOutType->glslType);
    keys.push_back(tailMulType->glslType);
    replacements.emplace_back("%tail_out_t%", tailOutType->glslType);
    replacements.emplace_back("%tail_out_t_lowest%", tailOutType->lowest);
    replacements.emplace_back("%tail_out_t_max%", tailOutType->max);
    replacements.emplace_back("%tail_mul_t%", tailMulType->glslType);
}

SpecConstants tileConstants(const RescaleTail *tail, const Conv2DTiles &tiles) {
    SpecConstants constants = rescaleTailConstants(tail);
    if (tiles.inputWords != 0) {
        constants.resize(5, 0);
        constants.push_back(tiles.inputWords);
        constants.push_back(tiles.weightWords);
        constants.push_back(tiles.groups);
        constants.push_back(tiles.groups);
    }
    return constants;
}

} // namespace

/*******************************************************************************
 * Conv2D
 *******************************************************************************/

Conv2D::Conv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_weights,
               const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<int32_t> &_pad,
               const std::vector<int32_t> &_stride, const std::vector<int32_t> &_dilation, const int8_t _inputZeroPoint,
               const int8_t _weightZeroPoint, const uint32_t _accType, const std::array<uint32_t, 3> &_maxGroupCount,
               const std::string &debugName, const RescaleTail *_tail, const Conv2DTiles &_tiles)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _weights, _biases, _tail),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _weights, _accType, _tail, _tiles), debugName,
                      tileConstants(_tail, _tiles)),
      pushConstant{createPushConstant(_pad, _stride, _dilation, _inputZeroPoint, _weightZeroPoint)},
      maxGroupCount{_maxGroupCount}, tiles{_tiles} {}

Conv2D::PushConstant Conv2D::createPushConstant(const std::vector<int32_t> &pad, const std::vector<int32_t> &stride,
                                                const std::vector<int32_t> &dilation, const int8_t inputZeroPoint,
                                                const int8_t weightZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        weightZeroPoint,
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
        },
        {
            stride[0],
            stride[1],
        },
        {
            dilation[0],
            dilation[1],
        },
        {0, 0, 0},
    };

    return constant;
}

DescriptorMap Conv2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output,
                                          const std::shared_ptr<TensorDescriptor> &weights,
                                          const std::shared_ptr<TensorDescriptor> &biases,
                                          const RescaleTail *tail) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, weights}, // set 2
        {Input, biases},  // set 3
    };

    appendRescaleTail(descriptorMap, tail);

    return descriptorMap;
}

SpirvBinary Conv2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output,
                                const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType,
                                const RescaleTail *tail, const Conv2DTiles &tiles) const {
    const auto *inType = getFormatInfo(input->getFormat());
    const auto *outType = getFormatInfo(tail != nullptr ? tail->inputFormat : output->getFormat());
    const auto *weightType = getFormatInfo(weights->getFormat());
    const auto *accTypeType = getFormatInfo(accTypeVkFormat(accType));

    const std::string warpXValue = std::to_string(warpX);
    const std::string warpYValue = std::to_string(warpY);
    const std::string warpZValue = std::to_string(warpZ);
    const std::string pixelsValue = std::to_string(tiles.pixels);
    const std::string pixelsKey = "px" + pixelsValue;

    PipelineCache::KeyList keys = {
        inType->glslType,
        weightType->glslType,
        outType->glslType,
        accTypeType->glslType,
    };
    PipelineCache::ReplaceList replacements = {
        {"%warpX%", warpXValue},
        {"%warpY%", warpYValue},
        {"%warpZ%", warpZValue},
        {"%in_t%", inType->glslType},
        {"%in_t_type%", inType->typeId},
        {"%out_t%", outType->glslType},
        {"%out_t_type%", outType->typeId},
        {"%weight_t%", weightType->glslType},
        {"%weight_t_type%", weightType->typeId},
        {"%acc_t_type%", accTypeType->typeId},
        {"%acc_t%", accTypeType->glslType},
    };
    appendRescaleTailReplacements(keys, replacements, output, tail);

    if (tiles.inputWords != 0 && tiles.dot) {
        keys.push_back(pixelsKey);
        replacements.emplace_back("%tile_pixels%", pixelsValue);
        return _pipelineCache->lookup(tileDotShaderName, keys, replacements);
    }

    const std::string_view tiledName = inType->isInteger ? tileShaderName : tileFloatShaderName;
    return _pipelineCache->lookup(tiles.inputWords != 0 ? tiledName : shaderName, keys, replacements);
}

void Conv2D::getTileGroupCounts(const std::shared_ptr<TensorDescriptor> &output, const uint32_t pixels,
                                uint32_t &groupCountX, uint32_t &groupCountY) {
    const auto &dimensions = output->getDimensions();
    groupCountX = divideRoundUp(static_cast<uint32_t>(dimensions[2]), warpX * pixels);
    groupCountY = divideRoundUp(static_cast<uint32_t>(dimensions[1]), warpY);
}

bool Conv2D::getTileWords(const std::shared_ptr<TensorDescriptor> &input,
                          const std::shared_ptr<TensorDescriptor> &weights, const std::vector<int32_t> &stride,
                          const std::vector<int32_t> &dilation, const uint32_t groups, const uint32_t sharedMemoryBytes,
                          const uint32_t valuesPerWord, const uint32_t pixels, uint32_t &inputWords,
                          uint32_t &weightWords) {
    if (input->getRank() != 4 || weights->getRank() != 4 || stride.size() != 2 || dilation.size() != 2 ||
        stride[0] < 1 || stride[1] < 1 || dilation[0] < 1 || dilation[1] < 1 ||
        (valuesPerWord != 1 && valuesPerWord != 4)) {
        return false;
    }

    const auto &weightDimensions = weights->getDimensions();
    const uint64_t wordBytes = 4;
    const uint64_t words =
        (static_cast<uint64_t>(input->getDimensions()[3]) + 3) / 4 * (4 / static_cast<uint64_t>(valuesPerWord));
    const uint64_t tileHeight = (warpY - 1) * static_cast<uint64_t>(stride[0]) +
                                (static_cast<uint64_t>(weightDimensions[1]) - 1) * static_cast<uint64_t>(dilation[0]) +
                                1;
    const uint64_t tileWidth = (warpX * static_cast<uint64_t>(pixels) - 1) * static_cast<uint64_t>(stride[1]) +
                               (static_cast<uint64_t>(weightDimensions[2]) - 1) * static_cast<uint64_t>(dilation[1]) +
                               1;
    const uint64_t tileInputWords = tileHeight * tileWidth * words;
    const uint64_t tileWeightWords = 4 * static_cast<uint64_t>(groups) * static_cast<uint64_t>(weightDimensions[1]) *
                                     static_cast<uint64_t>(weightDimensions[2]) * words;
    const uint64_t weightSumWords = valuesPerWord == 4 ? 4 * static_cast<uint64_t>(groups) : 0;
    if ((tileInputWords + tileWeightWords + weightSumWords) * wordBytes > sharedMemoryBytes) {
        return false;
    }

    inputWords = static_cast<uint32_t>(tileInputWords);
    weightWords = static_cast<uint32_t>(tileWeightWords);
    return true;
}

void Conv2D::cmdDispatch(VkCommandBuffer commandBuffer) {
    // Get first output tensor
    const auto &tensor = pipelineLayout->getTensorForSet(0);
    const auto &dimensions = tensor->getDimensions();
    if (tiles.inputWords != 0) {
        loader->vkCmdDispatch(commandBuffer, divideRoundUp(static_cast<uint32_t>(dimensions[2]), warpX * tiles.pixels),
                              divideRoundUp(static_cast<uint32_t>(dimensions[1]), warpY),
                              static_cast<uint32_t>(dimensions[0]) *
                                  divideRoundUp(divideRoundUp(static_cast<uint32_t>(dimensions[3]), 4), tiles.groups));
        return;
    }

    const auto totalGroupCountX = divideRoundUp(static_cast<uint32_t>(dimensions[0] * dimensions[2]), warpX);
    const auto totalGroupCountY = divideRoundUp(static_cast<uint32_t>(dimensions[1]), warpY);
    const auto totalGroupCountZ = divideRoundUp(static_cast<uint32_t>(dimensions[3]), warpZ * 4);

    for (uint32_t groupOffsetX = 0; groupOffsetX < totalGroupCountX; groupOffsetX += maxGroupCount[0]) {
        const auto groupCountX = std::min(maxGroupCount[0], totalGroupCountX - groupOffsetX);
        for (uint32_t groupOffsetY = 0; groupOffsetY < totalGroupCountY; groupOffsetY += maxGroupCount[1]) {
            const auto groupCountY = std::min(maxGroupCount[1], totalGroupCountY - groupOffsetY);
            for (uint32_t groupOffsetZ = 0; groupOffsetZ < totalGroupCountZ; groupOffsetZ += maxGroupCount[2]) {
                const auto groupCountZ = std::min(maxGroupCount[2], totalGroupCountZ - groupOffsetZ);
                auto dispatchPushConstant = pushConstant;
                dispatchPushConstant.outputPositionOffset[0] = groupOffsetX * warpX;
                dispatchPushConstant.outputPositionOffset[1] = groupOffsetY * warpY;
                dispatchPushConstant.outputPositionOffset[2] = groupOffsetZ * warpZ * 4;
                loader->vkCmdPushConstants(commandBuffer, pipelineLayout->getVkPipelineLayout(),
                                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispatchPushConstant),
                                           &dispatchPushConstant);
                loader->vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
            }
        }
    }
}

/*******************************************************************************
 * Conv3D
 *******************************************************************************/

Conv3D::Conv3D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_weights,
               const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<int32_t> &_pad,
               const std::vector<int32_t> &_stride, const std::vector<int32_t> &_dilation, const int8_t _inputZeroPoint,
               const int8_t _weightZeroPoint, const uint32_t _accType, const std::string &debugName,
               const RescaleTail *_tail)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _weights, _biases, _tail),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _weights, _accType, _tail), debugName,
                      rescaleTailConstants(_tail)),
      pushConstant{createPushConstant(_pad, _stride, _dilation, _inputZeroPoint, _weightZeroPoint)} {}

Conv3D::PushConstant Conv3D::createPushConstant(const std::vector<int32_t> &pad, const std::vector<int32_t> &stride,
                                                const std::vector<int32_t> &dilation, const int8_t inputZeroPoint,
                                                const int8_t weightZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        weightZeroPoint,
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
            pad[4],
            pad[5],
        },
        {
            stride[0],
            stride[1],
            stride[2],
        },
        {
            dilation[0],
            dilation[1],
            dilation[2],
        },
    };

    return constant;
}

DescriptorMap Conv3D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output,
                                          const std::shared_ptr<TensorDescriptor> &weights,
                                          const std::shared_ptr<TensorDescriptor> &biases,
                                          const RescaleTail *tail) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, weights}, // set 2
        {Input, biases},  // set 3
    };

    appendRescaleTail(descriptorMap, tail);

    return descriptorMap;
}

SpirvBinary Conv3D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output,
                                const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType,
                                const RescaleTail *tail) const {
    const auto *inType = getFormatInfo(input->getFormat());
    const auto *outType = getFormatInfo(tail != nullptr ? tail->inputFormat : output->getFormat());
    const auto *weightType = getFormatInfo(weights->getFormat());
    const auto *accTypeType = getFormatInfo(accTypeVkFormat(accType));

    PipelineCache::KeyList keys = {
        inType->glslType,
        weightType->glslType,
        outType->glslType,
        accTypeType->glslType,
    };
    PipelineCache::ReplaceList replacements = {
        {"%warpX%", warp1DSv},
        {"%in_t%", inType->glslType},
        {"%in_t_type%", inType->typeId},
        {"%out_t%", outType->glslType},
        {"%out_t_type%", outType->typeId},
        {"%weight_t%", weightType->glslType},
        {"%weight_t_type%", weightType->typeId},
        {"%acc_t_type%", accTypeType->typeId},
        {"%acc_t%", accTypeType->glslType},
    };
    appendRescaleTailReplacements(keys, replacements, output, tail);

    return _pipelineCache->lookup(shaderName, keys, replacements);
}

/*******************************************************************************
 * DepthwiseConv2D
 *******************************************************************************/

DepthwiseConv2D::DepthwiseConv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                 VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &_input,
                                 const std::shared_ptr<TensorDescriptor> &_output,
                                 const std::shared_ptr<TensorDescriptor> &_weights,
                                 const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<int32_t> &_pad,
                                 const std::vector<int32_t> &_stride, const std::vector<int32_t> &_dilation,
                                 const int8_t _inputZeroPoint, const int8_t _weightZeroPoint, const uint32_t _accType,
                                 const std::string &debugName, const RescaleTail *_tail)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _weights, _biases, _tail),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _weights, _accType, _tail), debugName,
                      rescaleTailConstants(_tail)),
      pushConstant{createPushConstant(_pad, _stride, _dilation, _inputZeroPoint, _weightZeroPoint)} {}

DepthwiseConv2D::PushConstant DepthwiseConv2D::createPushConstant(const std::vector<int32_t> &pad,
                                                                  const std::vector<int32_t> &stride,
                                                                  const std::vector<int32_t> &dilation,
                                                                  const int8_t inputZeroPoint,
                                                                  const int8_t weightZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        weightZeroPoint,
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
        },
        {
            stride[0],
            stride[1],
        },
        {
            dilation[0],
            dilation[1],
        },
    };

    return constant;
}

DescriptorMap DepthwiseConv2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                                   const std::shared_ptr<TensorDescriptor> &output,
                                                   const std::shared_ptr<TensorDescriptor> &weights,
                                                   const std::shared_ptr<TensorDescriptor> &biases,
                                                   const RescaleTail *tail) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, weights}, // set 2
        {Input, biases},  // set 3
    };

    appendRescaleTail(descriptorMap, tail);

    return descriptorMap;
}

SpirvBinary DepthwiseConv2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                         const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output,
                                         const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType,
                                         const RescaleTail *tail) const {
    const auto *inType = getFormatInfo(input->getFormat());
    const auto *outType = getFormatInfo(tail != nullptr ? tail->inputFormat : output->getFormat());
    const auto *weightType = getFormatInfo(weights->getFormat());
    const auto *accTypeType = getFormatInfo(accTypeVkFormat(accType));

    PipelineCache::KeyList keys = {
        inType->glslType,
        weightType->glslType,
        outType->glslType,
        accTypeType->glslType,
    };
    PipelineCache::ReplaceList replacements = {
        {"%warpX%", warp1DSv},
        {"%in_t%", inType->glslType},
        {"%in_t_type%", inType->typeId},
        {"%out_t%", outType->glslType},
        {"%out_t_type%", outType->typeId},
        {"%weight_t%", weightType->glslType},
        {"%weight_t_type%", weightType->typeId},
        {"%acc_t_type%", accTypeType->typeId},
        {"%acc_t%", accTypeType->glslType},
    };
    appendRescaleTailReplacements(keys, replacements, output, tail);

    return _pipelineCache->lookup(shaderName, keys, replacements);
}

void DepthwiseConv2D::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * ElementwiseBinary
 *******************************************************************************/

ElementwiseBinary::ElementwiseBinary(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
    const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
    const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
    const uint32_t _nanMode, const std::string &debugName, const std::string_view &_operation)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _output),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input1, _output, debugName, _operation), debugName,
                      {_input1->getRank(), _output->getRank()}),
      pushConstant{createPushConstant(_nanMode)} {}

ElementwiseBinary::PushConstant ElementwiseBinary::createPushConstant(const uint32_t nanMode) const {
    return PushConstant{nanMode};
}

DescriptorMap ElementwiseBinary::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                                     const std::shared_ptr<TensorDescriptor> &input2,
                                                     const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
    };

    return descriptorMap;
}

SpirvBinary ElementwiseBinary::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                           const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &output, const std::string &name,
                                           const std::string_view &operation) const {
    const auto *inType = getFormatInfo(input->getFormat());
    const auto *outType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      name,
                                      inType->glslType,
                                      outType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%operation%", operation},
                                      {"%in_t%", inType->glslType},
                                      {"%out_t%", outType->glslType},
                                      {"%in_t_type%", inType->typeId},
                                      {"%out_t_type%", outType->typeId},
                                      {"%in_t_comp%", inType->compType},
                                  });
}

void ElementwiseBinary::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * ElementwiseUnary
 *******************************************************************************/

ElementwiseUnary::ElementwiseUnary(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                   VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                                   const std::shared_ptr<TensorDescriptor> &_input1,
                                   const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName,
                                   const std::string_view &_operation)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _output, debugName, _operation), debugName, {_output->getRank()}) {}

DescriptorMap ElementwiseUnary::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                                    const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
    };

    return descriptorMap;
}

SpirvBinary ElementwiseUnary::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                          const std::shared_ptr<TensorDescriptor> &output, const std::string &name,
                                          const std::string_view &operation) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      name,
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%operation%", operation},
                                      {"%in_out_t%", inOutType->glslType},
                                      {"%in_out_t_type%", inOutType->typeId},
                                      {"%in_out_t_comp%", inOutType->compType},
                                  });
}

void ElementwiseUnary::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Fft2D
 *******************************************************************************/

Fft2D::Fft2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
             const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_inputReal,
             const std::shared_ptr<TensorDescriptor> &_inputImag, const std::shared_ptr<TensorDescriptor> &_outputReal,
             const std::shared_ptr<TensorDescriptor> &_outputImag, const bool _inverse, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_inputReal, _inputImag, _outputReal, _outputImag),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache, createSpirv(_pipelineCache), debugName),
      pushConstant{createPushConstant(_inverse)} {}

Fft2D::PushConstant Fft2D::createPushConstant(const bool inverse) const {
    float signValue = inverse ? -1.0f : 1.0f;
    PushConstant constant = {
        signValue,
    };
    return constant;
}

DescriptorMap Fft2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &inputReal,
                                         const std::shared_ptr<TensorDescriptor> &inputImag,
                                         const std::shared_ptr<TensorDescriptor> &outputReal,
                                         const std::shared_ptr<TensorDescriptor> &outputImag) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, outputReal}, // set 0
        {Output, outputImag}, // set 1
        {Input, inputReal},   // set 2
        {Input, inputImag},   // set 3
    };

    return descriptorMap;
}

SpirvBinary Fft2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache) const {
    return _pipelineCache->lookup(shaderName, {},
                                  {
                                      {"%warpX%", warp1DSv},
                                  });
}

/*******************************************************************************
 * Gather
 *******************************************************************************/

Gather::Gather(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_values,
               const std::shared_ptr<TensorDescriptor> &_indices, const std::shared_ptr<TensorDescriptor> &_output,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_values, _indices, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _indices, _output), debugName, {}) {}

DescriptorMap Gather::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &values,
                                          const std::shared_ptr<TensorDescriptor> &indices,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, values},  // set 1
        {Input, indices}, // set 2
    };

    return descriptorMap;
}

SpirvBinary Gather::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &indices,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());
    const auto *indicesType = getFormatInfo(indices->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                      indicesType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%index_t%", indicesType->glslType},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

/*******************************************************************************
 * Matmul
 *******************************************************************************/

Matmul::Matmul(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
               const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
               const int32_t _inputZeroPoint1, const int32_t _inputZeroPoint2, const std::string &debugName,
               const RescaleTail *_tail)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _output, _tail),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input1, _output, _tail), debugName, rescaleTailConstants(_tail)),
      pushConstant{createPushConstant(_inputZeroPoint1, _inputZeroPoint2)} {}

Matmul::PushConstant Matmul::createPushConstant(const int32_t inputZeroPoint1, const int32_t inputZeroPoint2) const {
    PushConstant constant = {
        inputZeroPoint1,
        inputZeroPoint2,
    };

    return constant;
}

DescriptorMap Matmul::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                          const std::shared_ptr<TensorDescriptor> &input2,
                                          const std::shared_ptr<TensorDescriptor> &output,
                                          const RescaleTail *tail) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
    };

    appendRescaleTail(descriptorMap, tail);

    return descriptorMap;
}

SpirvBinary Matmul::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input1,
                                const std::shared_ptr<TensorDescriptor> &output, const RescaleTail *tail) const {
    const auto *inType = getFormatInfo(input1->getFormat());
    const auto *outType = getFormatInfo(tail != nullptr ? tail->inputFormat : output->getFormat());

    PipelineCache::KeyList keys = {
        inType->glslType,
        outType->glslType,
    };
    PipelineCache::ReplaceList replacements = {
        {"%warpX%", warp1DSv},          {"%in_t%", inType->glslType},      {"%in_t_type%", inType->typeId},
        {"%out_t%", outType->glslType}, {"%out_t_type%", outType->typeId},
    };
    appendRescaleTailReplacements(keys, replacements, output, tail);

    return _pipelineCache->lookup(shaderName, keys, replacements);
}

void Matmul::cmdDispatch(VkCommandBuffer commandBuffer) {
    const auto *inType = getFormatInfo(pipelineLayout->getTensorForSet(1)->getFormat());
    cmdDispatchVector(commandBuffer, 0, inType->isInteger ? 4u : 1u);
}

/*******************************************************************************
 * MaxPool2D
 *******************************************************************************/

MaxPool2D::MaxPool2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                     const std::shared_ptr<TensorDescriptor> &_input, const std::shared_ptr<TensorDescriptor> &_output,
                     const std::vector<int32_t> &_kernel, const std::vector<int32_t> &_stride,
                     const std::vector<int32_t> &_pad, const uint32_t _nanMode, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output, _nanMode), debugName),
      pushConstant{createPushConstant(_kernel, _stride, _pad, _nanMode)} {}

MaxPool2D::PushConstant MaxPool2D::createPushConstant(const std::vector<int32_t> &kernel,
                                                      const std::vector<int32_t> &stride,
                                                      const std::vector<int32_t> &pad, const uint32_t nanMode) const {
    PushConstant constant = {
        {
            kernel[0],
            kernel[1],
        },
        {
            stride[0],
            stride[1],
        },
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
        },
        nanMode,
    };

    return constant;
}

DescriptorMap MaxPool2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                             const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary MaxPool2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                   const std::shared_ptr<TensorDescriptor> &output, const uint32_t _nanMode) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    const std::string_view init = (_nanMode == NanPropagationMode::Ignore ? "NAN" : inOutType->lowest);
    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                      {"%in_out_t_lowest%", init},
                                      {"%in_out_t_type%", inOutType->typeId},
                                      {"%in_out_t_comp%", inOutType->compType},
                                  });
}

void MaxPool2D::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Mul
 *******************************************************************************/

Mul::Mul(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
         const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
         const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
         const uint32_t _shift, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _output),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input1, _output), debugName,
                      {_input1->getRank(), _output->getRank()}),
      pushConstant{createPushConstant(_shift)} {}

Mul::PushConstant Mul::createPushConstant(const uint32_t shift) const {
    PushConstant constant = {
        shift,
    };

    return constant;
}

DescriptorMap Mul::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                       const std::shared_ptr<TensorDescriptor> &input2,
                                       const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
    };

    return descriptorMap;
}

SpirvBinary Mul::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                             const std::shared_ptr<TensorDescriptor> &input1,
                             const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inType = getFormatInfo(input1->getFormat());
    const auto *outType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType,
                                      outType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_t_type%", inType->typeId},
                                      {"%out_t_type%", outType->typeId},
                                      {"%in_t%", inType->glslType},
                                      {"%out_t%", outType->glslType},
                                  });
}

/*******************************************************************************
 * Negate
 *******************************************************************************/

Negate::Negate(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const int32_t _inputZeroPoint,
               const int32_t _outputZeroPoint, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output), debugName, {_output->getRank()}),
      pushConstant{createPushConstant(_inputZeroPoint, _outputZeroPoint)} {}

Negate::PushConstant Negate::createPushConstant(const int32_t inputZeroPoint, const int32_t outputZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        outputZeroPoint,
    };

    return constant;
}

DescriptorMap Negate::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Negate::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    const std::string_view accType = inOutType->isInteger ? "int32_t" : "float";

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                      {"%in_out_t_type%", inOutType->typeId},
                                      {"%acc_t%", accType},
                                      {"%in_out_t_lowest%", inOutType->lowest},
                                      {"%in_out_t_max%", inOutType->max},
                                  });
}

void Negate::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Pad
 *******************************************************************************/

Pad::Pad(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
         const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
         const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_padding,
         const real_t _padConst, const int32_t _padConstInt, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _padding),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache, createSpirv(_pipelineCache, _output),
                      debugName, {_input->getRank()}),
      pushConstant{createPushConstant(_padConst, _padConstInt)} {}

Pad::PushConstant Pad::createPushConstant(const real_t padConst, const int32_t padConstInt) const {
    PushConstant constant = {
        padConst,
        padConstInt,
    };

    return constant;
}

DescriptorMap Pad::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                       const std::shared_ptr<TensorDescriptor> &output,
                                       const std::shared_ptr<TensorDescriptor> &padding) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, padding}, // set 2
    };

    return descriptorMap;
}

SpirvBinary Pad::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                             const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                      {"%in_out_t_type%", inOutType->typeId},
                                  });
}

void Pad::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Reduce
 *******************************************************************************/

Reduce::Reduce(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _nanMode,
               const std::string &debugName, const std::string &_init, const std::string_view &_operation)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _input, debugName, _init, _operation), debugName,
                      {_output->getRank()}),
      pushConstant{createPushConstant(_axis, _nanMode, getFormatInfo(_input->getFormat())->isInteger)} {}

Reduce::PushConstant Reduce::createPushConstant(const uint32_t axis, const uint32_t nanMode,
                                                const bool isInteger) const {
    PushConstant constant = {
        axis,
        isInteger ? static_cast<uint32_t>(NanPropagationMode::Propagate) : nanMode, // Enforce propagate for integers
    };

    return constant;
}

DescriptorMap Reduce::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Reduce::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &name,
                                const std::string &init, const std::string_view &operation) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      name,
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%init%", init},
                                      {"%operation%", operation},
                                      {"%in_out_t%", inOutType->glslType},
                                      {"%in_out_t_type%", inOutType->typeId},
                                      {"%in_out_t_comp%", inOutType->compType},
                                  });
}

/*******************************************************************************
 * Rescale
 *******************************************************************************/

Rescale::Rescale(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
                 const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
                 const std::shared_ptr<TensorDescriptor> &_output, const int32_t _inputZeroPoint,
                 const int32_t _outputZeroPoint, const std::shared_ptr<TensorDescriptor> &_multiplier,
                 const std::shared_ptr<TensorDescriptor> &_shift, const bool _scale32, const bool _doubleRound,
                 const bool _perChannel, const bool _inputUnsigned, const bool _outputUnsigned,
                 const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _multiplier, _shift),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _multiplier, _inputUnsigned, _outputUnsigned),
                      debugName, {_output->getRank(), _scale32, _doubleRound, _perChannel}),
      pushConstant{createPushConstant(_inputZeroPoint, _outputZeroPoint)} {}

Rescale::PushConstant Rescale::createPushConstant(const int32_t inputZeroPoint, const int32_t outputZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        outputZeroPoint,
    };

    return constant;
}

DescriptorMap Rescale::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &output,
                                           const std::shared_ptr<TensorDescriptor> &multiplier,
                                           const std::shared_ptr<TensorDescriptor> &shift) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output},    // set 0
        {Input, input},      // set 1
        {Input, multiplier}, // set 2
        {Input, shift},      // set 3
    };

    return descriptorMap;
}

SpirvBinary Rescale::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &input,
                                 const std::shared_ptr<TensorDescriptor> &output,
                                 const std::shared_ptr<TensorDescriptor> &multiplier, const bool inputUnsigned,
                                 const bool outputUnsigned) const {
    const auto *inType = getFormatInfo(input->getFormat(), inputUnsigned);
    const auto *outType = getFormatInfo(output->getFormat(), outputUnsigned);
    const auto *mulType = getFormatInfo(multiplier->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType,
                                      outType->glslType,
                                      mulType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_t%", inType->glslType},
                                      {"%out_t%", outType->glslType},
                                      {"%mul_t%", mulType->glslType},
                                      {"%out_t_lowest%", outType->lowest},
                                      {"%out_t_max%", outType->max},
                                  });
}

/*******************************************************************************
 * Reshape
 *******************************************************************************/

Reshape::Reshape(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
                 const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
                 const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _output), debugName, {_input->getRank(), _output->getRank()}) {}

DescriptorMap Reshape::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Reshape::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

/*******************************************************************************
 * Resize
 *******************************************************************************/

Resize::Resize(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const std::vector<int32_t> &_scale,
               const std::vector<int32_t> &_offset, const std::vector<int32_t> &_border, const uint32_t _mode,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _input, _output), debugName),
      pushConstant{createPushConstant(_scale, _offset, _border, _mode)} {}

Resize::PushConstant Resize::createPushConstant(const std::vector<int32_t> &scale, const std::vector<int32_t> &offset,
                                                const std::vector<int32_t> &border, const uint32_t mode) const {
    PushConstant constant = {
        {
            scale[0],
            scale[1],
            scale[2],
            scale[3],
        },
        {
            offset[0],
            offset[1],
        },
        {
            border[0],
            border[1],
        },
        mode,
    };

    return constant;
}

DescriptorMap Resize::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

void Resize::cmdDispatch(VkCommandBuffer commandBuffer) {
    const VkFormat format = pipelineLayout->getTensorForSet(0)->getFormat();
    const bool scalarFloat = format == VK_FORMAT_R16_SFLOAT || format == VK_FORMAT_R32_SFLOAT ||
                             format == VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM;
    cmdDispatchVector(commandBuffer, 0, scalarFloat ? 1u : 4u);
}

SpirvBinary Resize::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inType = getFormatInfo(input->getFormat());
    const auto *outType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType,
                                      outType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_t%", inType->glslType},
                                      {"%in_t_type%", inType->typeId},
                                      {"%out_t%", outType->glslType},
                                      {"%out_t_type%", outType->typeId},
                                      {"%out_t_comp%", outType->compType},
                                  });
}

/*******************************************************************************
 * Reverse
 *******************************************************************************/

Reverse::Reverse(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
                 const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
                 const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output), debugName, {_output->getRank()}),
      pushConstant{createPushConstant(_axis)} {}

Reverse::PushConstant Reverse::createPushConstant(const uint32_t axis) const {
    PushConstant constant = {
        axis,
    };

    return constant;
}

DescriptorMap Reverse::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &output) const {
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Reverse::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

void Reverse::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Rfft2D
 *******************************************************************************/

Rfft2D::Rfft2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_outputReal,
               const std::shared_ptr<TensorDescriptor> &_outputImag, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _outputReal, _outputImag), {}, _pipelineCache,
                      createSpirv(_pipelineCache), debugName) {}

DescriptorMap Rfft2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &outputReal,
                                          const std::shared_ptr<TensorDescriptor> &outputImag) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, outputReal}, // set 0
        {Output, outputImag}, // set 1
        {Input, input},       // set 2
    };

    return descriptorMap;
}

SpirvBinary Rfft2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache) const {
    return _pipelineCache->lookup(shaderName, {},
                                  {
                                      {"%warpX%", warp1DSv},
                                  });
}

/*******************************************************************************
 * Scatter
 *******************************************************************************/

Scatter::Scatter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
                 const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
                 const std::shared_ptr<TensorDescriptor> &_values, const std::shared_ptr<TensorDescriptor> &_indices,
                 const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _values, _indices, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _indices, _output), debugName, {}) {}

DescriptorMap Scatter::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &values,
                                           const std::shared_ptr<TensorDescriptor> &indices,
                                           const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, values},  // set 2
        {Input, indices}, // set 3
    };

    return descriptorMap;
}

SpirvBinary Scatter::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &indices,
                                 const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());
    const auto *indicesType = getFormatInfo(indices->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                      indicesType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%index_t%", indicesType->glslType},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

/*******************************************************************************
 * Select
 *******************************************************************************/

Select::Select(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
               const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_input3,
               const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _input3, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _output), debugName, {_output->getRank()}) {}

DescriptorMap Select::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                          const std::shared_ptr<TensorDescriptor> &input2,
                                          const std::shared_ptr<TensorDescriptor> &input3,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
        {Input, input3},  // set 3
    };

    return descriptorMap;
}

SpirvBinary Select::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

/*******************************************************************************
 * Slice
 *******************************************************************************/

Slice::Slice(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
             const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
             const std::shared_ptr<TensorDescriptor> &_output, const std::vector<uint32_t> &_start,
             const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output),
                      {&pushConstant, static_cast<uint32_t>(_input->getRank() * sizeof(uint32_t))}, _pipelineCache,
                      createSpirv(_pipelineCache, _input), debugName, {_input->getRank()}),
      pushConstant{createPushConstant(_start)} {}

Slice::PushConstant Slice::createPushConstant(const std::vector<uint32_t> &start) const {
    PushConstant constant{};
    std::copy(start.begin(), start.end(), &constant.start[0]);
    return constant;
}

DescriptorMap Slice::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Slice::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                               const std::shared_ptr<TensorDescriptor> &input) const {
    const auto *inOutType = getFormatInfo(input->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

void Slice::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Table
 *******************************************************************************/

Table::Table(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
             const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
             const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_table,
             const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _table), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output), debugName, {_output->getRank()}) {}

DescriptorMap Table::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output,
                                         const std::shared_ptr<TensorDescriptor> &table) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, table},   // set 2
    };

    return descriptorMap;
}

SpirvBinary Table::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                               const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inType = getFormatInfo(input->getFormat());
    const auto *outType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType,
                                      outType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_t%", inType->glslType},
                                      {"%out_t%", outType->glslType},
                                  });
}

void Table::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Tile
 *******************************************************************************/

Tile::Tile(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _output), debugName, {_input->getRank()}) {}

DescriptorMap Tile::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                        const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Tile::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                              const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

void Tile::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * Transpose
 *******************************************************************************/

Transpose::Transpose(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                     const std::shared_ptr<TensorDescriptor> &_input, const std::shared_ptr<TensorDescriptor> &_output,
                     const std::vector<uint32_t> &_perms, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output),
                      {&pushConstant, static_cast<uint32_t>(_input->getRank() * sizeof(uint32_t))}, _pipelineCache,
                      createSpirv(_pipelineCache, _output), debugName, {_output->getRank()}),
      pushConstant{createPushConstant(_perms)} {}

Transpose::PushConstant Transpose::createPushConstant(const std::vector<uint32_t> &perms) const {
    PushConstant constant{};
    std::copy(perms.begin(), perms.end(), &constant.perms[0]);
    return constant;
}

DescriptorMap Transpose::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                             const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Transpose::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                   const std::shared_ptr<TensorDescriptor> &output) const {
    const auto *inOutType = getFormatInfo(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType,
                                  },
                                  {
                                      {"%warpX%", warp1DSv},
                                      {"%in_out_t%", inOutType->glslType},
                                  });
}

void Transpose::cmdDispatch(VkCommandBuffer commandBuffer) { cmdDispatchVector(commandBuffer, 0, 4); }

/*******************************************************************************
 * TransposeConv2D
 *******************************************************************************/

TransposeConv2D::TransposeConv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                 VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &_input,
                                 const std::shared_ptr<TensorDescriptor> &_output,
                                 const std::shared_ptr<TensorDescriptor> &_weights,
                                 const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<int32_t> &_outPad,
                                 const std::vector<int32_t> &_stride, const int8_t _inputZeroPoint,
                                 const int8_t _weightZeroPoint, const uint32_t _accType, const std::string &debugName,
                                 const RescaleTail *_tail)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _weights, _biases, _tail),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _weights, _accType, _tail), debugName,
                      rescaleTailConstants(_tail)),
      pushConstant{createPushConstant(_outPad, _stride, _inputZeroPoint, _weightZeroPoint)} {}

TransposeConv2D::PushConstant TransposeConv2D::createPushConstant(const std::vector<int32_t> &outPad,
                                                                  const std::vector<int32_t> &stride,
                                                                  const int8_t inputZeroPoint,
                                                                  const int8_t weightZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        weightZeroPoint,
        {
            outPad[0],
            outPad[1],
            outPad[2],
            outPad[3],
        },
        {
            stride[0],
            stride[1],
        },
    };

    return constant;
}

DescriptorMap TransposeConv2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                                   const std::shared_ptr<TensorDescriptor> &output,
                                                   const std::shared_ptr<TensorDescriptor> &weights,
                                                   const std::shared_ptr<TensorDescriptor> &biases,
                                                   const RescaleTail *tail) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, weights}, // set 2
        {Input, biases},  // set 3
    };

    appendRescaleTail(descriptorMap, tail);

    return descriptorMap;
}

SpirvBinary TransposeConv2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                         const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output,
                                         const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType,
                                         const RescaleTail *tail) const {
    const auto *inType = getFormatInfo(input->getFormat());
    const auto *outType = getFormatInfo(tail != nullptr ? tail->inputFormat : output->getFormat());
    const auto *weightType = getFormatInfo(weights->getFormat());
    const auto *accTypeType = getFormatInfo(accTypeVkFormat(accType));

    PipelineCache::KeyList keys = {
        inType->glslType,
        weightType->glslType,
        outType->glslType,
        accTypeType->glslType,
    };
    PipelineCache::ReplaceList replacements = {
        {"%warpX%", warp1DSv},
        {"%in_t%", inType->glslType},
        {"%in_t_type%", inType->typeId},
        {"%out_t%", outType->glslType},
        {"%out_t_type%", outType->typeId},
        {"%weight_t%", weightType->glslType},
        {"%weight_t_type%", weightType->typeId},
        {"%acc_t_type%", accTypeType->typeId},
        {"%acc_t%", accTypeType->glslType},
    };
    appendRescaleTailReplacements(keys, replacements, output, tail);

    return _pipelineCache->lookup(shaderName, keys, replacements);
}

/*******************************************************************************
 * BlockMatch
 *******************************************************************************/

BlockMatch::BlockMatch(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                       VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                       const std::shared_ptr<TensorDescriptor> &inTemplate,
                       const std::shared_ptr<TensorDescriptor> &inSearch,
                       const std::optional<std::shared_ptr<TensorDescriptor>> &outVectors,
                       const std::optional<std::shared_ptr<TensorDescriptor>> &outCosts,
                       const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
                       const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
                       const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding,
                       const uint32_t searchPattern, const SearchType searchType, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(inTemplate, inSearch, outVectors, outCosts, searchType), {},
                      _pipelineCache, createSpirv(_pipelineCache, searchType), debugName,
                      createSpecConstants(kernelSizes, searchWindowSizes, inputStrides, windowStrides, windowOffsets,
                                          padding, searchPattern)) {}

DescriptorMap BlockMatch::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &inTemplate,
                                              const std::shared_ptr<TensorDescriptor> &inSearch,
                                              const std::optional<std::shared_ptr<TensorDescriptor>> &outVectors,
                                              const std::optional<std::shared_ptr<TensorDescriptor>> &outCosts,
                                              const SearchType searchType) const {

    DescriptorMap descriptorMap = {
        {Input, inTemplate}, // set 0
        {Input, inSearch},   // set 1
    };

    switch (searchType) {
    case SearchType::MIN_SAD: {
        assert(outVectors);
        descriptorMap.emplace_back(Output, outVectors.value()); // set 2
        break;
    }
    case SearchType::RAW_SAD: {
        assert(outCosts);
        descriptorMap.emplace_back(Output, outCosts.value()); // set 2
        break;
    }
    case SearchType::MIN_SAD_COST: {
        assert(outVectors && outCosts);
        descriptorMap.emplace_back(Output, outVectors.value()); // set 2
        descriptorMap.emplace_back(Output, outCosts.value());   // set 3
        break;
    }
    }

    return descriptorMap;
}

SpirvBinary BlockMatch::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                    const SearchType searchType) const {
    const auto searchTypeStr = std::to_string(static_cast<uint32_t>(searchType));
    return _pipelineCache->lookup(shaderName, {searchTypeStr},
                                  {
                                      {"%search_type%", searchTypeStr},
                                      {"%warpX%", std::to_string(warpX)},
                                      {"%warpY%", std::to_string(warpY)},
                                  });
}

SpecConstants BlockMatch::createSpecConstants(const std::vector<uint32_t> &kernelSizes,
                                              const std::vector<uint32_t> &searchWindowSizes,
                                              const std::vector<uint32_t> &inputStrides,
                                              const std::vector<uint32_t> &windowStrides,
                                              const std::vector<uint32_t> &windowOffsets,
                                              const std::vector<uint32_t> &padding,
                                              const uint32_t searchPattern) const {
    SpecConstants specConstants = {
        kernelSizes[0],  kernelSizes[1],   searchWindowSizes[0], searchWindowSizes[1], inputStrides[0],
        inputStrides[1], windowStrides[0], windowStrides[1],     windowOffsets[0],     windowOffsets[1],
        padding[0],      padding[1],       searchPattern,
    };

    return specConstants;
}

void BlockMatch::cmdDispatch(VkCommandBuffer commandBuffer) {
    const auto &tensor = pipelineLayout->getTensorForSet(0);
    const auto &dimensions = tensor->getDimensions();
    loader->vkCmdDispatch(commandBuffer, divideRoundUp(static_cast<uint32_t>(dimensions[3]), warpX),
                          divideRoundUp(static_cast<uint32_t>(dimensions[2]), warpY), 1);
}

/*******************************************************************************
 * GraphPipeline
 *******************************************************************************/

GraphPipeline::GraphPipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                             VkPhysicalDevice _physicalDevice, VkDevice _device,
                             const std::shared_ptr<PipelineCache> &_pipelineCache)
    : loader{_loader}, physicalDevice{_physicalDevice}, device{_device}, pipelineCache{_pipelineCache} {
    VkPhysicalDeviceProperties properties{};
    loader->vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    maxComputeWorkGroupCount = {
        properties.limits.maxComputeWorkGroupCount[0],
        properties.limits.maxComputeWorkGroupCount[1],
        properties.limits.maxComputeWorkGroupCount[2],
    };
    maxComputeWorkGroupSize = {
        properties.limits.maxComputeWorkGroupSize[0],
        properties.limits.maxComputeWorkGroupSize[1],
        properties.limits.maxComputeWorkGroupSize[2],
    };
    maxComputeWorkGroupInvocations = properties.limits.maxComputeWorkGroupInvocations;
    maxComputeSharedMemorySize = properties.limits.maxComputeSharedMemorySize;
}

GraphPipeline::~GraphPipeline() {
    for (auto &deviceMemory : constantsDeviceMemory) {
        loader->vkFreeMemory(device, deviceMemory, nullptr);
    }
}

void GraphPipeline::makeConstTensor(const uint32_t id, const VkTensorDescriptionARM &tensorDescription,
                                    const void *data) {
    const auto tensorDescriptor = std::make_shared<TensorDescriptor>(loader, physicalDevice, device, tensorDescription);
    auto tensor = TensorDescriptor::makeTensor(tensorDescriptor);
    auto *const deviceMemory = tensorDescriptor->createInitializeDeviceMemory(data);

    (void)tensor->bindTensorMemory(deviceMemory, 0);
    constantsDeviceMemory.push_back(deviceMemory);
    constTensorMap[id] = std::move(tensor);
}

std::shared_ptr<TensorDescriptor> GraphPipeline::getConstTensor(const uint32_t id) const {
    return constTensorMap.at(id)->getTensorDescriptor();
}

std::shared_ptr<TensorDescriptor>
GraphPipeline::makeConstCompositeTensor(const VkFormat format, std::vector<int64_t> dimensions, const void *data) {
    auto tensorDescriptor =
        std::make_shared<TensorDescriptor>(loader, physicalDevice, device, format, std::move(dimensions));
    auto tensor = TensorDescriptor::makeTensor(tensorDescriptor);
    auto *const deviceMemory = tensorDescriptor->createInitializeDeviceMemory(data);

    (void)tensor->bindTensorMemory(deviceMemory, 0);
    constantsDeviceMemory.push_back(deviceMemory);
    compositeTensors.emplace_back(std::move(tensor));

    return tensorDescriptor;
}

ComputeDescriptorSetMap GraphPipeline::makeConstantsDescriptorSets() const {
    TensorDescriptorMap filter;

    for ([[maybe_unused]] const auto &[_, tensor] : constTensorMap) {
        filter[tensor->getTensorDescriptor()] = tensor;
    }

    for (const auto &tensor : compositeTensors) {
        filter[tensor->getTensorDescriptor()] = tensor;
    }

    return getComputeDescriptorSetMap(filter);
}

bool GraphPipeline::hasIntegerDotProduct() {
    if (integerDotProduct < 0) {
        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &vulkan13Features;
        loader->vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

        const char *const disabled = std::getenv("VMEL_DISABLE_CONV_DOT");
        integerDotProduct = vulkan13Features.shaderIntegerDotProduct == VK_TRUE &&
                                    (disabled == nullptr || std::string_view(disabled) == "0")
                                ? 1
                                : 0;
    }

    return integerDotProduct == 1;
}

void GraphPipeline::makeDescriptorSetBinding(const uint32_t set, const uint32_t binding, const uint32_t arrayIndex,
                                             const VkTensorDescriptionARM &tensorDescription) {
    auto tensorDescriptor = std::make_shared<TensorDescriptor>(loader, physicalDevice, device, tensorDescription);

    auto &vec = tensorMap[set][binding];
    vec.resize(std::max(vec.size(), size_t(arrayIndex + 1)));
    vec[arrayIndex] = tensorDescriptor;

    tensorDescriptorMap[set][tensorDescriptor] = TensorDescriptor::makeTensor(tensorDescriptor);
}

std::shared_ptr<TensorDescriptor> GraphPipeline::getTensor(const uint32_t set, const uint32_t binding,
                                                           const uint32_t arrayIndex) const {
    return tensorMap.at(set).at(binding).at(arrayIndex);
}

std::shared_ptr<TensorDescriptor> GraphPipeline::makeTensor(const VkFormat format, std::vector<int64_t> dimensions,
                                                            std::vector<int64_t> strides) {
    auto tensor = std::make_shared<TensorDescriptor>(loader, physicalDevice, device, format, std::move(dimensions),
                                                     std::move(strides));

    auto [iterator, inserted] = tensorSet.insert(tensor);

    if (inserted) {
        tensors.push_back(tensor);
    }

    return tensor;
}

const std::vector<std::shared_ptr<TensorDescriptor>> &GraphPipeline::getTensors() const { return tensors; }

ComputeDescriptorSetMap GraphPipeline::makeSessionRamDescriptorSets() const {
    TensorDescriptorMap filter;
    for (const auto &tensorDescriptor : tensorSet) {
        filter[tensorDescriptor] = TensorDescriptor::makeTensor(tensorDescriptor);
    }

    return getComputeDescriptorSetMap(filter);
}

ComputeDescriptorSetMap GraphPipeline::makeExternalDescriptorSets(const uint32_t set) const {
    const auto filterIt = tensorDescriptorMap.find(set);
    if (filterIt == tensorDescriptorMap.end()) {
        return {};
    }

    return getComputeDescriptorSetMap(filterIt->second);
}

ComputeDescriptorSetMap GraphPipeline::getComputeDescriptorSetMap(const TensorDescriptorMap &filter) const {
    ComputeDescriptorSetMap mapping;
    for (const auto &pipeline : pipelines) {
        const auto &pipelineLayout = pipeline->getComputePipelineLayout();
        pipelineLayout->makeDescriptorSets(mapping, filter);
    }
    return mapping;
}

void GraphPipeline::cmdBindAndDispatch(VkCommandBuffer commandBuffer, const ComputeDescriptorSetMap &descriptorSetMap,
                                       const ComputePipelineDispatchDecorator &dispatchDecorator) {
    for (uint32_t i = 0; i < pipelines.size(); ++i) {
        if (dispatchDecorator) {
            dispatchDecorator(commandBuffer, *pipelines[i], descriptorSetMap, i);
        } else {
            pipelines[i]->cmdBindAndDispatch(commandBuffer, descriptorSetMap);
        }
    }
}

const std::vector<std::shared_ptr<ComputePipelineBase>> &GraphPipeline::getPipelines() const { return pipelines; }

void GraphPipeline::makeInput(const std::shared_ptr<TensorDescriptor> &tensor) {
    // Register inputs pipeline as producer of tensors
    tensor->setPipeline(&inputs);
}

void GraphPipeline::makeOutput(const std::shared_ptr<TensorDescriptor> &tensor) {
    // Connect outputs pipeline with parent pipelines
    makeAndConnectVirtualTensor(tensor, &outputs);
}

/*******************************************************************************
 * Tosa Ops
 *******************************************************************************/

void GraphPipeline::makeAbs(const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input, output, debugName, "abs(value1)");
}

void GraphPipeline::makeAdd(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &input2,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 + value2");
}

void GraphPipeline::makeArgmax(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                               const uint32_t nanMode, const std::string &debugName) {
    makePipeline<Argmax>(input, output, axis, nanMode, debugName);
}

void GraphPipeline::makeArithmeticRightShift(const std::shared_ptr<TensorDescriptor> &input1,
                                             const std::shared_ptr<TensorDescriptor> &input2,
                                             const std::shared_ptr<TensorDescriptor> &output, const bool round,
                                             const std::string &debugName) {
    makePipeline<ArithmeticRightShift>(input1, input2, output, round, debugName);
}

void GraphPipeline::makeAvgPool2D(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::vector<int32_t> &kernel,
                                  const std::vector<int32_t> &stride, const std::vector<int32_t> &pad,
                                  const uint32_t accType, const int8_t inputZeroPoint, const int8_t outputZeroPoint,
                                  const std::string &debugName) {
    makePipeline<AvgPool2D>(input, output, kernel, stride, pad, accType, inputZeroPoint, outputZeroPoint, debugName);
}

void GraphPipeline::makeBitwiseAnd(const std::shared_ptr<TensorDescriptor> &input1,
                                   const std::shared_ptr<TensorDescriptor> &input2,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 & value2");
}

void GraphPipeline::makeBitwiseNot(const std::shared_ptr<TensorDescriptor> &input,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input, output, debugName, "~value1");
}

void GraphPipeline::makeBitwiseOr(const std::shared_ptr<TensorDescriptor> &input1,
                                  const std::shared_ptr<TensorDescriptor> &input2,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 | value2");
}

void GraphPipeline::makeBitwiseXor(const std::shared_ptr<TensorDescriptor> &input1,
                                   const std::shared_ptr<TensorDescriptor> &input2,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 ^ value2");
}

void GraphPipeline::makeCast(const std::shared_ptr<TensorDescriptor> &input,
                             const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<Cast>(input, output, debugName);
}

void GraphPipeline::makeCeil(const std::shared_ptr<TensorDescriptor> &input1,
                             const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input1, output, debugName, "ceil(value1)");
}

void GraphPipeline::makeClamp(const std::shared_ptr<TensorDescriptor> &input,
                              const std::shared_ptr<TensorDescriptor> &output, const real_t min, const real_t max,
                              const uint32_t nanMode, const std::string &debugName) {
    makePipeline<Clamp>(input, output, min, max, nanMode, debugName);
}

void GraphPipeline::makeClz(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input1, output, debugName, "clz(value1)");
}

void GraphPipeline::makeConcat(const std::vector<std::shared_ptr<TensorDescriptor>> &_inputs,
                               const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                               const std::string &debugName) {
    uint32_t offset = 0;
    for (const auto &input : _inputs) {
        makePipeline<Concat>(input, output, axis, offset, debugName);
        offset += static_cast<uint32_t>(input->getDimensions()[axis]);
    }
}

Conv2DTiles GraphPipeline::selectConv2DTiles(const std::shared_ptr<TensorDescriptor> &input,
                                             const std::shared_ptr<TensorDescriptor> &output,
                                             const std::shared_ptr<TensorDescriptor> &weights,
                                             const std::vector<int32_t> &stride, const std::vector<int32_t> &dilation,
                                             const uint32_t accType, const RescaleTail *tail) {
    uint32_t tileInputWords = 0;
    uint32_t tileWeightWords = 0;
    uint32_t tileGroups = 1;
    uint32_t tilePixels = 1;
    static const bool disableTiles = []() {
        const char *const value = std::getenv("VMEL_DISABLE_CONV_TILES");
        return value != nullptr && std::string_view(value) != "0";
    }();
    const VkFormat sumFormat = tail != nullptr ? tail->inputFormat : output->getFormat();
    const bool int8Tiles = input->getFormat() == VK_FORMAT_R8_SINT && weights->getFormat() == VK_FORMAT_R8_SINT &&
                           accTypeVkFormat(accType) == VK_FORMAT_R32_SINT && sumFormat == VK_FORMAT_R32_SINT;
    const bool floatTiles = tail == nullptr && input->getFormat() == VK_FORMAT_R32_SFLOAT &&
                            weights->getFormat() == VK_FORMAT_R32_SFLOAT &&
                            accTypeVkFormat(accType) == VK_FORMAT_R32_SFLOAT && sumFormat == VK_FORMAT_R32_SFLOAT;
    if (!disableTiles && (int8Tiles || floatTiles) && output->getRank() == 4) {
        const auto &outputDimensions = output->getDimensions();
        constexpr uint64_t maxTileGroups = 4;
        const uint64_t channelGroups = (static_cast<uint64_t>(outputDimensions[3]) + 3) / 4;
        const uint32_t wantedPixels = int8Tiles && hasIntegerDotProduct() ? 2u : 1u;
        bool fits = false;
        for (uint32_t pixelsTry = wantedPixels; !fits; pixelsTry /= 2) {
            uint32_t tileGroupCountX = 0;
            uint32_t tileGroupCountY = 0;
            Conv2D::getTileGroupCounts(output, pixelsTry, tileGroupCountX, tileGroupCountY);
            for (uint32_t groupsTry =
                     static_cast<uint32_t>(channelGroups < maxTileGroups ? channelGroups : maxTileGroups);
                 groupsTry >= 1; groupsTry /= 2) {
                const uint64_t workgroupsZ =
                    static_cast<uint64_t>(outputDimensions[0]) * ((channelGroups + groupsTry - 1) / groupsTry);
                if (Conv2D::warpX * Conv2D::warpY * groupsTry <= maxComputeWorkGroupInvocations &&
                    groupsTry <= maxComputeWorkGroupSize[2] && workgroupsZ <= maxComputeWorkGroupCount[2] &&
                    tileGroupCountX <= maxComputeWorkGroupCount[0] && tileGroupCountY <= maxComputeWorkGroupCount[1] &&
                    Conv2D::getTileWords(input, weights, stride, dilation, groupsTry, maxComputeSharedMemorySize,
                                         int8Tiles ? 4u : 1u, pixelsTry, tileInputWords, tileWeightWords)) {
                    tileGroups = groupsTry;
                    tilePixels = pixelsTry;
                    fits = true;
                    break;
                }
                if (groupsTry == 1u) {
                    break;
                }
            }
            if (fits || pixelsTry == 1u) {
                break;
            }
        }
        if (!fits) {
            tileInputWords = 0;
            tileWeightWords = 0;
            tileGroups = 1;
            tilePixels = 1;
        }
    }
    Conv2DTiles tiles;
    tiles.inputWords = tileInputWords;
    tiles.weightWords = tileWeightWords;
    tiles.groups = tileGroups;
    tiles.pixels = tilePixels;
    tiles.dot = tileInputWords != 0 && int8Tiles && hasIntegerDotProduct();
    return tiles;
}

void GraphPipeline::makeConv2D(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output,
                               const std::shared_ptr<TensorDescriptor> &weights,
                               const std::shared_ptr<TensorDescriptor> &biases, const std::vector<int32_t> &pad,
                               const std::vector<int32_t> &stride, const std::vector<int32_t> &dilation,
                               const int8_t inputZeroPoint, const int8_t weightZeroPoint, const uint32_t accType,
                               const std::string &debugName, const RescaleTail *tail) {
    makePipeline<Conv2D>(input, output, weights, biases, pad, stride, dilation, inputZeroPoint, weightZeroPoint,
                         accType, maxComputeWorkGroupCount, debugName, tail,
                         selectConv2DTiles(input, output, weights, stride, dilation, accType, tail));
}

void GraphPipeline::makeConv3D(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output,
                               const std::shared_ptr<TensorDescriptor> &weights,
                               const std::shared_ptr<TensorDescriptor> &biases, const std::vector<int32_t> &pad,
                               const std::vector<int32_t> &stride, const std::vector<int32_t> &dilation,
                               const int8_t inputZeroPoint, const int8_t weightZeroPoint, const uint32_t accType,
                               const std::string &debugName, const RescaleTail *tail) {
    makePipeline<Conv3D>(input, output, weights, biases, pad, stride, dilation, inputZeroPoint, weightZeroPoint,
                         accType, debugName, tail);
}

void GraphPipeline::makeCos(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input1, output, debugName, "cos(value1)");
}

void GraphPipeline::makeDepthwiseConv2D(const std::shared_ptr<TensorDescriptor> &input,
                                        const std::shared_ptr<TensorDescriptor> &output,
                                        const std::shared_ptr<TensorDescriptor> &weights,
                                        const std::shared_ptr<TensorDescriptor> &biases,
                                        const std::vector<int32_t> &pad, const std::vector<int32_t> &stride,
                                        const std::vector<int32_t> &dilation, const int8_t inputZeroPoint,
                                        const int8_t weightZeroPoint, const uint32_t accType,
                                        const std::string &debugName, const RescaleTail *tail) {
    makePipeline<DepthwiseConv2D>(input, output, weights, biases, pad, stride, dilation, inputZeroPoint,
                                  weightZeroPoint, accType, debugName, tail);
}

void GraphPipeline::makeEqual(const std::shared_ptr<TensorDescriptor> &input1,
                              const std::shared_ptr<TensorDescriptor> &input2,
                              const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 == value2");
}

void GraphPipeline::makeErf(const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input, output, debugName, "erf(value1)");
}

void GraphPipeline::makeExp(const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input, output, debugName, "exp(value1)");
}

void GraphPipeline::makeFft2D(const std::shared_ptr<TensorDescriptor> &inputReal,
                              const std::shared_ptr<TensorDescriptor> &inputImag,
                              const std::shared_ptr<TensorDescriptor> &outputReal,
                              const std::shared_ptr<TensorDescriptor> &outputImag, const bool inverse,
                              const std::string &debugName) {
    makePipeline<Fft2D>(inputReal, inputImag, outputReal, outputImag, inverse, debugName);
}

void GraphPipeline::makeFloor(const std::shared_ptr<TensorDescriptor> &input1,
                              const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input1, output, debugName, "floor(value1)");
}

void GraphPipeline::makeGather(const std::shared_ptr<TensorDescriptor> &values,
                               const std::shared_ptr<TensorDescriptor> &indices,
                               const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<Gather>(values, indices, output, debugName);
}

void GraphPipeline::makeGreater(const std::shared_ptr<TensorDescriptor> &input1,
                                const std::shared_ptr<TensorDescriptor> &input2,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 > value2");
}

void GraphPipeline::makeGreaterEqual(const std::shared_ptr<TensorDescriptor> &input1,
                                     const std::shared_ptr<TensorDescriptor> &input2,
                                     const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 >= value2");
}

void GraphPipeline::makeIntdiv(const std::shared_ptr<TensorDescriptor> &input1,
                               const std::shared_ptr<TensorDescriptor> &input2,
                               const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 / value2");
}

void GraphPipeline::makeLog(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input1, output, debugName, "log_guarded(value1)");
}

void GraphPipeline::makeLogicalAnd(const std::shared_ptr<TensorDescriptor> &input1,
                                   const std::shared_ptr<TensorDescriptor> &input2,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 && value2");
}

void GraphPipeline::makeLogicalLeftShift(const std::shared_ptr<TensorDescriptor> &input1,
                                         const std::shared_ptr<TensorDescriptor> &input2,
                                         const std::shared_ptr<TensorDescriptor> &output,
                                         const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "uint(value1) << uint(value2)");
}

void GraphPipeline::makeLogicalNot(const std::shared_ptr<TensorDescriptor> &input,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input, output, debugName, "!value1");
}

void GraphPipeline::makeLogicalRightShift(const std::shared_ptr<TensorDescriptor> &input1,
                                          const std::shared_ptr<TensorDescriptor> &input2,
                                          const std::shared_ptr<TensorDescriptor> &output,
                                          const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "zeroExtend(value1) >> uint(value2)");
}

void GraphPipeline::makeLogicalOr(const std::shared_ptr<TensorDescriptor> &input1,
                                  const std::shared_ptr<TensorDescriptor> &input2,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 || value2");
}

void GraphPipeline::makeLogicalXor(const std::shared_ptr<TensorDescriptor> &input1,
                                   const std::shared_ptr<TensorDescriptor> &input2,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 ^^ value2");
}

void GraphPipeline::makeMatmul(const std::shared_ptr<TensorDescriptor> &input1,
                               const std::shared_ptr<TensorDescriptor> &input2,
                               const std::shared_ptr<TensorDescriptor> &output, const int32_t inputZeroPoint1,
                               const int32_t inputZeroPoint2, const std::string &debugName, const RescaleTail *tail) {
    makePipeline<Matmul>(input1, input2, output, inputZeroPoint1, inputZeroPoint2, debugName, tail);
}

void GraphPipeline::makeMaxPool2D(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::vector<int32_t> &kernel,
                                  const std::vector<int32_t> &stride, const std::vector<int32_t> &pad,
                                  const uint32_t nanMode, const std::string &debugName) {
    makePipeline<MaxPool2D>(input, output, kernel, stride, pad, nanMode, debugName);
}

void GraphPipeline::makeMaximum(const std::shared_ptr<TensorDescriptor> &input1,
                                const std::shared_ptr<TensorDescriptor> &input2,
                                const std::shared_ptr<TensorDescriptor> &output, const uint32_t nanMode,
                                const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, nanMode, debugName,
                                    "applyMax(value1, value2, pushConstants.nanMode)");
}

void GraphPipeline::makeMinimum(const std::shared_ptr<TensorDescriptor> &input1,
                                const std::shared_ptr<TensorDescriptor> &input2,
                                const std::shared_ptr<TensorDescriptor> &output, const uint32_t nanMode,
                                const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, nanMode, debugName,
                                    "applyMin(value1, value2, pushConstants.nanMode)");
}

void GraphPipeline::makeMul(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &input2,
                            const std::shared_ptr<TensorDescriptor> &output, const uint32_t shift,
                            const std::string &debugName) {
    makePipeline<Mul>(input1, input2, output, shift, debugName);
}

void GraphPipeline::makeNegate(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output, const int32_t inputZeroPoint,
                               const int32_t outputZeroPoint, const std::string &debugName) {
    makePipeline<Negate>(input, output, inputZeroPoint, outputZeroPoint, debugName);
}

void GraphPipeline::makePad(const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output,
                            const std::shared_ptr<TensorDescriptor> &padding, const real_t padConst,
                            const int32_t padConstInt, const std::string &debugName) {
    makePipeline<Pad>(input, output, padding, padConst, padConstInt, debugName);
}

void GraphPipeline::makePow(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &input2,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "power(value1, value2)");
}

void GraphPipeline::makeReciprocal(const std::shared_ptr<TensorDescriptor> &input,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input, output, debugName, "1.0 / value1");
}

void GraphPipeline::makeReduceAll(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const std::string &debugName) {
    makePipeline<Reduce>(input, output, axis, NanPropagationMode::Propagate, debugName, "true", "result && value");
}

void GraphPipeline::makeReduceAny(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const std::string &debugName) {
    makePipeline<Reduce>(input, output, axis, NanPropagationMode::Propagate, debugName, "false", "result || value");
}

void GraphPipeline::makeReduceMax(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const uint32_t nanMode, const std::string &debugName) {
    const auto *inOutType = getFormatInfo(output->getFormat());
    const std::string init =
        "(pushConstants.nanMode == NAN_MODE_IGNORE) ? IN_OUT_T(NAN) : IN_OUT_T(" + std::string(inOutType->lowest) + ')';
    makePipeline<Reduce>(input, output, axis, nanMode, debugName, init, "max(result, value)");
}

void GraphPipeline::makeReduceMin(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const uint32_t nanMode, const std::string &debugName) {
    const auto *inOutType = getFormatInfo(output->getFormat());
    const std::string init =
        "(pushConstants.nanMode == NAN_MODE_IGNORE) ? IN_OUT_T(NAN) : IN_OUT_T(" + std::string(inOutType->max) + ')';
    makePipeline<Reduce>(input, output, axis, nanMode, debugName, init, "min(result, value)");
}

void GraphPipeline::makeReduceProduct(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                      const std::string &debugName) {
    makePipeline<Reduce>(input, output, axis, NanPropagationMode::Propagate, debugName, "1", "result * value");
}

void GraphPipeline::makeReduceSum(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const std::string &debugName) {
    makePipeline<Reduce>(input, output, axis, NanPropagationMode::Propagate, debugName, "0", "result + value");
}

void GraphPipeline::makeRescale(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output, const int32_t inputZeroPoint,
                                const int32_t outputZeroPoint, const std::shared_ptr<TensorDescriptor> &multiplier,
                                const std::shared_ptr<TensorDescriptor> &shift, const bool scale32,
                                const bool doubleRound, const bool perChannel, const bool inputUnsigned,
                                const bool outputUnsigned, const std::string &debugName) {
    makePipeline<Rescale>(input, output, inputZeroPoint, outputZeroPoint, multiplier, shift, scale32, doubleRound,
                          perChannel, inputUnsigned, outputUnsigned, debugName);
}

void GraphPipeline::makeReshape(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<Reshape>(input, output, debugName);
}

void GraphPipeline::makeResize(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output, const std::vector<int32_t> &scale,
                               const std::vector<int32_t> &offset, const std::vector<int32_t> &border,
                               const uint32_t mode, const std::string &debugName) {
    makePipeline<Resize>(input, output, scale, offset, border, mode, debugName);
}

void GraphPipeline::makeReverse(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                const std::string &debugName) {
    makePipeline<Reverse>(input, output, axis, debugName);
}

void GraphPipeline::makeRfft2D(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &outputReal,
                               const std::shared_ptr<TensorDescriptor> &outputImag, const std::string &debugName) {
    makePipeline<Rfft2D>(input, outputReal, outputImag, debugName);
}

void GraphPipeline::makeRsqrt(const std::shared_ptr<TensorDescriptor> &input1,
                              const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input1, output, debugName, "inversesqrt(value1)");
}

void GraphPipeline::makeScatter(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &values,
                                const std::shared_ptr<TensorDescriptor> &indices,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<Scatter>(input, values, indices, output, debugName);
}

void GraphPipeline::makeSelect(const std::shared_ptr<TensorDescriptor> &input1,
                               const std::shared_ptr<TensorDescriptor> &input2,
                               const std::shared_ptr<TensorDescriptor> &input3,
                               const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<Select>(input1, input2, input3, output, debugName);
}

void GraphPipeline::makeSigmoid(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input, output, debugName, "1.0 / (1.0 + exp(-value1))");
}

void GraphPipeline::makeSin(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input1, output, debugName, "sin_hybrid(value1)");
}

void GraphPipeline::makeSlice(const std::shared_ptr<TensorDescriptor> &input,
                              const std::shared_ptr<TensorDescriptor> &output, const std::vector<uint32_t> &start,
                              const std::string &debugName) {
    makePipeline<Slice>(input, output, start, debugName);
}

void GraphPipeline::makeSub(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &input2,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseBinary>(input1, input2, output, NanPropagationMode::Propagate, debugName,
                                    "value1 - value2");
}

void GraphPipeline::makeTable(const std::shared_ptr<TensorDescriptor> &input,
                              const std::shared_ptr<TensorDescriptor> &output,
                              const std::shared_ptr<TensorDescriptor> &table, const std::string &debugName) {
    makePipeline<Table>(input, output, table, debugName);
}

void GraphPipeline::makeTanh(const std::shared_ptr<TensorDescriptor> &input1,
                             const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<ElementwiseUnary>(input1, output, debugName, "tanh_clamped(value1)");
}

void GraphPipeline::makeTile(const std::shared_ptr<TensorDescriptor> &input,
                             const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    makePipeline<Tile>(input, output, debugName);
}

void GraphPipeline::makeTranspose(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::vector<uint32_t> &perms,
                                  const std::string &debugName) {
    makePipeline<Transpose>(input, output, perms, debugName);
}

void GraphPipeline::makeTransposeConv2D(const std::shared_ptr<TensorDescriptor> &input,
                                        const std::shared_ptr<TensorDescriptor> &output,
                                        const std::shared_ptr<TensorDescriptor> &weights,
                                        const std::shared_ptr<TensorDescriptor> &biases,
                                        const std::vector<int32_t> &pad, const std::vector<int32_t> &stride,
                                        const int8_t inputZeroPoint, const int8_t weightZeroPoint,
                                        const uint32_t accType, const std::string &debugName, const RescaleTail *tail) {
    makePipeline<TransposeConv2D>(input, output, weights, biases, pad, stride, inputZeroPoint, weightZeroPoint, accType,
                                  debugName, tail);
}

/*******************************************************************************
 * Motion Engine Ops
 *******************************************************************************/

void GraphPipeline::makeMinSadCost(
    const std::shared_ptr<TensorDescriptor> &inTemplate, const std::shared_ptr<TensorDescriptor> &inSearch,
    const std::shared_ptr<TensorDescriptor> &outVectors, const std::shared_ptr<TensorDescriptor> &outCosts,
    const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
    const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
    const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding, const uint32_t searchPattern,
    const std::string &debugName) {
    makePipeline<BlockMatch>(inTemplate, inSearch, outVectors, outCosts, kernelSizes, searchWindowSizes, inputStrides,
                             windowStrides, windowOffsets, padding, searchPattern, BlockMatch::SearchType::MIN_SAD_COST,
                             debugName);
}

void GraphPipeline::makeMinSad(const std::shared_ptr<TensorDescriptor> &inTemplate,
                               const std::shared_ptr<TensorDescriptor> &inSearch,
                               const std::shared_ptr<TensorDescriptor> &outVectors,
                               const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
                               const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
                               const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding,
                               const uint32_t searchPattern, const std::string &debugName) {
    makePipeline<BlockMatch>(inTemplate, inSearch, outVectors, std::nullopt, kernelSizes, searchWindowSizes,
                             inputStrides, windowStrides, windowOffsets, padding, searchPattern,
                             BlockMatch::SearchType::MIN_SAD, debugName);
}

void GraphPipeline::makeRawSad(const std::shared_ptr<TensorDescriptor> &inTemplate,
                               const std::shared_ptr<TensorDescriptor> &inSearch,
                               const std::shared_ptr<TensorDescriptor> &outCosts,
                               const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
                               const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
                               const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding,
                               const std::string &debugName) {
    makePipeline<BlockMatch>(inTemplate, inSearch, std::nullopt, outCosts, kernelSizes, searchWindowSizes, inputStrides,
                             windowStrides, windowOffsets, padding, 0, BlockMatch::SearchType::RAW_SAD, debugName);
}

} // namespace graph_op
} // namespace mlsdk::el::compute
