#include "runtime/function/render/include/render/vulkan_manager/vulkan_util.h"
#include "render/vulkan_manager/vulkan_render_pass.h"
#include "runtime/function/render/include/render/vulkan_manager/vulkan_context.h"

#include "vulkan/vulkan_core.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

std::unordered_map<uint32_t, VkSampler> Pilot::PVulkanUtil::m_mipmap_sampler_map;
VkSampler                               Pilot::PVulkanUtil::m_nearest_sampler = VK_NULL_HANDLE;
VkSampler                               Pilot::PVulkanUtil::m_linear_sampler  = VK_NULL_HANDLE;

uint32_t Pilot::PVulkanUtil::findMemoryType(VkPhysicalDevice      physical_device,
                                            uint32_t              type_filter,
                                            VkMemoryPropertyFlags properties_flag)
{
    VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_device_memory_properties);
    for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; i++)
    {
        if (type_filter & (1 << i) &&
            (physical_device_memory_properties.memoryTypes[i].propertyFlags & properties_flag) == properties_flag)
        {
            return i;
        }
    }
    throw std::runtime_error("findMemoryType");
}

VkShaderModule Pilot::PVulkanUtil::createShaderModule(VkDevice device, const std::vector<unsigned char>& shader_code)
{
    VkShaderModuleCreateInfo shader_module_create_info {};
    shader_module_create_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_create_info.codeSize = shader_code.size();
    shader_module_create_info.pCode    = reinterpret_cast<const uint32_t*>(shader_code.data());

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device, &shader_module_create_info, nullptr, &shader_module) != VK_SUCCESS)
    {
        return VK_NULL_HANDLE;
    }
    return shader_module;
}

void Pilot::PVulkanUtil::createBuffer(VkPhysicalDevice      physical_device,
                                      VkDevice              device,
                                      VkDeviceSize          size,
                                      VkBufferUsageFlags    usage,
                                      VkMemoryPropertyFlags properties,
                                      VkBuffer&             buffer,
                                      VkDeviceMemory&       buffer_memory,
                                      void*                 data)
{
    VkBufferCreateInfo buffer_create_info {};
    buffer_create_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.size        = size;
    buffer_create_info.usage       = usage;                     // use as a vertex/staging/index buffer
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // not sharing among queue families

    if (vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateBuffer");
    }

    VkMemoryRequirements buffer_memory_requirements; // for allocate_info.allocationSize and
                                                     // allocate_info.memoryTypeIndex
    vkGetBufferMemoryRequirements(device, buffer, &buffer_memory_requirements);

    VkMemoryAllocateInfo buffer_memory_allocate_info {};
    buffer_memory_allocate_info.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    buffer_memory_allocate_info.allocationSize = buffer_memory_requirements.size;
    buffer_memory_allocate_info.memoryTypeIndex =
        PVulkanUtil::findMemoryType(physical_device, buffer_memory_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &buffer_memory_allocate_info, nullptr, &buffer_memory) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateMemory");
    }

    if (data)
    {
        void* mapped = nullptr;
        if (vkMapMemory(device, buffer_memory, 0, size, 0, &mapped))
        {
            throw std::runtime_error("vkMapMemory");
        }
        std::memcpy(mapped, data, size);
        if (mapped)
        {
            vkUnmapMemory(device, buffer_memory);
        }
    }

    // bind buffer with buffer memory
    vkBindBufferMemory(device, buffer, buffer_memory, 0); // offset = 0
}

void Pilot::PVulkanUtil::copyBuffer(PVulkanContext* context,
                                    VkBuffer        srcBuffer,
                                    VkBuffer        dstBuffer,
                                    VkDeviceSize    srcOffset,
                                    VkDeviceSize    dstOffset,
                                    VkDeviceSize    size)
{
    assert(context);

    VkCommandBuffer command_buffer = context->beginSingleTimeCommands();

    VkBufferCopy copy_region = {srcOffset, dstOffset, size};
    vkCmdCopyBuffer(command_buffer, srcBuffer, dstBuffer, 1, &copy_region);

    context->endSingleTimeCommands(command_buffer);
}

void Pilot::PVulkanUtil::createImage(VkPhysicalDevice      physical_device,
                                     VkDevice              device,
                                     uint32_t              image_width,
                                     uint32_t              image_height,
                                     VkFormat              format,
                                     VkImageTiling         image_tiling,
                                     VkImageUsageFlags     image_usage_flags,
                                     VkMemoryPropertyFlags memory_property_flags,
                                     VkImage&              image,
                                     VkDeviceMemory&       memory,
                                     VkImageCreateFlags    image_create_flags,
                                     uint32_t              array_layers,
                                     uint32_t              miplevels)
{
    VkImageCreateInfo image_create_info {};
    image_create_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.flags         = image_create_flags;
    image_create_info.imageType     = VK_IMAGE_TYPE_2D;
    image_create_info.extent.width  = image_width;
    image_create_info.extent.height = image_height;
    image_create_info.extent.depth  = 1;
    image_create_info.mipLevels     = miplevels;
    image_create_info.arrayLayers   = array_layers;
    image_create_info.format        = format;
    image_create_info.tiling        = image_tiling;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.usage         = image_usage_flags;
    image_create_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &image_create_info, nullptr, &image) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device, image, &mem_requirements);

    VkMemoryAllocateInfo alloc_info {};
    alloc_info.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
        findMemoryType(physical_device, mem_requirements.memoryTypeBits, memory_property_flags);

    if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, memory, 0);
}

VkImageView Pilot::PVulkanUtil::createImageView(VkDevice           device,
                                                VkImage&           image,
                                                VkFormat           format,
                                                VkImageAspectFlags image_aspect_flags,
                                                VkImageViewType    view_type,
                                                uint32_t           layout_count,
                                                uint32_t           miplevels)
{
    VkImageViewCreateInfo image_view_create_info {};
    image_view_create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.image                           = image;
    image_view_create_info.viewType                        = view_type;
    image_view_create_info.format                          = format;
    image_view_create_info.subresourceRange.aspectMask     = image_aspect_flags;
    image_view_create_info.subresourceRange.baseMipLevel   = 0;
    image_view_create_info.subresourceRange.levelCount     = miplevels;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount     = layout_count;

    VkImageView image_view;
    if (vkCreateImageView(device, &image_view_create_info, nullptr, &image_view) != VK_SUCCESS)
    {
        return image_view;
        // todo
    }

    return image_view;
}

void Pilot::PVulkanUtil::transitionImageLayout(PVulkanContext*    context,
                                               VkImage            image,
                                               VkImageLayout      old_layout,
                                               VkImageLayout      new_layout,
                                               uint32_t           layer_count,
                                               uint32_t           miplevels,
                                               VkImageAspectFlags aspect_mask_bits)
{
    assert(context);

    VkCommandBuffer command_buffer = context->beginSingleTimeCommands();

    VkImageMemoryBarrier barrier {};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = old_layout;
    barrier.newLayout                       = new_layout;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = image;
    barrier.subresourceRange.aspectMask     = aspect_mask_bits;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = miplevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = layer_count;

    VkPipelineStageFlags source_stage;
    VkPipelineStageFlags destination_stage;

    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        source_stage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        source_stage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    // for getGuidAndDepthOfMouseClickOnRenderSceneForUI() get depthimage
    else if (old_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        source_stage      = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        source_stage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    // for generating mipmapped image
    else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        source_stage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    context->endSingleTimeCommands(command_buffer);
}

void Pilot::PVulkanUtil::copyBufferToImage(PVulkanContext* context,
                                           VkBuffer        buffer,
                                           VkImage         image,
                                           uint32_t        width,
                                           uint32_t        height,
                                           uint32_t        layer_count)
{
    assert(context);

    VkCommandBuffer command_buffer = context->beginSingleTimeCommands();

    VkBufferImageCopy region {};
    region.bufferOffset                    = 0;
    region.bufferRowLength                 = 0;
    region.bufferImageHeight               = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = layer_count;
    region.imageOffset                     = {0, 0, 0};
    region.imageExtent                     = {width, height, 1};

    vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    context->endSingleTimeCommands(command_buffer);
}

void Pilot::PVulkanUtil::genMipmappedImage(PVulkanContext* context,
                                           VkImage         image,
                                           uint32_t        width,
                                           uint32_t        height,
                                           uint32_t        mip_levels)
{
    assert(context);

    VkCommandBuffer command_buffer = context->beginSingleTimeCommands();

    for (uint32_t i = 1; i < mip_levels; i++)
    {
        VkImageBlit image_blit {};
        image_blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_blit.srcSubresource.layerCount = 1;
        image_blit.srcSubresource.mipLevel   = i - 1;
        image_blit.srcOffsets[1].x           = std::max(static_cast<int32_t>(width >> (i - 1)), 1);
        image_blit.srcOffsets[1].y           = std::max(static_cast<int32_t>(height >> (i - 1)), 1);
        image_blit.srcOffsets[1].z           = 1;

        image_blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_blit.dstSubresource.layerCount = 1;
        image_blit.dstSubresource.mipLevel   = i;
        image_blit.dstOffsets[1].x           = std::max(static_cast<int32_t>(width >> i), 1);
        image_blit.dstOffsets[1].y           = std::max(static_cast<int32_t>(height >> i), 1);
        image_blit.dstOffsets[1].z           = 1;

        VkImageSubresourceRange mip_sub_range {};
        mip_sub_range.aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT;
        mip_sub_range.baseMipLevel = i;
        mip_sub_range.levelCount   = 1;
        mip_sub_range.layerCount   = 1;

        VkImageMemoryBarrier barrier {};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcAccessMask       = 0;
        barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image;
        barrier.subresourceRange    = mip_sub_range;

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier);

        vkCmdBlitImage(command_buffer,
                       image,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &image_blit,
                       VK_FILTER_LINEAR);

        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier);
    }

    VkImageSubresourceRange mip_sub_range {};
    mip_sub_range.aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT;
    mip_sub_range.baseMipLevel = 0;
    mip_sub_range.levelCount   = mip_levels;
    mip_sub_range.layerCount   = 1;

    VkImageMemoryBarrier barrier {};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = mip_sub_range;

    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);

    context->endSingleTimeCommands(command_buffer);
}

VkSampler Pilot::PVulkanUtil::getOrCreateMipmapSampler(VkPhysicalDevice physical_device,
                                                       VkDevice         device,
                                                       uint32_t         width,
                                                       uint32_t         height)
{
    assert(width > 0 && height > 0);

    VkSampler sampler;
    uint32_t  mip_levels   = floor(log2(std::max(width, height))) + 1;
    auto      find_sampler = m_mipmap_sampler_map.find(mip_levels);
    if (find_sampler != m_mipmap_sampler_map.end())
    {
        return find_sampler->second;
    }
    else
    {
        VkPhysicalDeviceProperties physical_device_properties {};
        vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

        VkSamplerCreateInfo sampler_info {};
        sampler_info.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter    = VK_FILTER_LINEAR;
        sampler_info.minFilter    = VK_FILTER_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        sampler_info.anisotropyEnable = VK_TRUE;
        sampler_info.maxAnisotropy    = physical_device_properties.limits.maxSamplerAnisotropy;

        sampler_info.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sampler_info.unnormalizedCoordinates = VK_FALSE;
        sampler_info.compareEnable           = VK_FALSE;
        sampler_info.compareOp               = VK_COMPARE_OP_ALWAYS;
        sampler_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        sampler_info.maxLod = mip_levels - 1;

        if (vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS)
        {
            assert(0);
        }
    }

    m_mipmap_sampler_map.insert(std::make_pair(mip_levels, sampler));

    return sampler;
}

void Pilot::PVulkanUtil::destroyMipmappedSampler(VkDevice device)
{
    for (auto sampler : m_mipmap_sampler_map)
    {
        vkDestroySampler(device, sampler.second, nullptr);
    }
    m_mipmap_sampler_map.clear();
}

VkSampler Pilot::PVulkanUtil::getOrCreateNearestSampler(VkPhysicalDevice physical_device, VkDevice device)
{
    if (m_nearest_sampler == VK_NULL_HANDLE)
    {
        VkPhysicalDeviceProperties physical_device_properties {};
        vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

        VkSamplerCreateInfo sampler_info {};

        sampler_info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter               = VK_FILTER_NEAREST;
        sampler_info.minFilter               = VK_FILTER_NEAREST;
        sampler_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.mipLodBias              = 0.0f;
        sampler_info.anisotropyEnable        = VK_FALSE;
        sampler_info.maxAnisotropy           = physical_device_properties.limits.maxSamplerAnisotropy; // close :1.0f
        sampler_info.compareEnable           = VK_FALSE;
        sampler_info.compareOp               = VK_COMPARE_OP_ALWAYS;
        sampler_info.minLod                  = 0.0f;
        sampler_info.maxLod                  = 8.0f; // todo: m_irradiance_texture_miplevels
        sampler_info.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sampler_info.unnormalizedCoordinates = VK_FALSE;

        if (vkCreateSampler(device, &sampler_info, nullptr, &m_nearest_sampler) != VK_SUCCESS)
        {
            throw std::runtime_error("vk create sampler");
        }
    }

    return m_nearest_sampler;
}

VkSampler Pilot::PVulkanUtil::getOrCreateLinearSampler(VkPhysicalDevice physical_device, VkDevice device)
{
    if (m_linear_sampler == VK_NULL_HANDLE)
    {
        VkPhysicalDeviceProperties physical_device_properties {};
        vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

        VkSamplerCreateInfo sampler_info {};

        sampler_info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter               = VK_FILTER_LINEAR;
        sampler_info.minFilter               = VK_FILTER_LINEAR;
        sampler_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.mipLodBias              = 0.0f;
        sampler_info.anisotropyEnable        = VK_FALSE;
        sampler_info.maxAnisotropy           = physical_device_properties.limits.maxSamplerAnisotropy; // close :1.0f
        sampler_info.compareEnable           = VK_FALSE;
        sampler_info.compareOp               = VK_COMPARE_OP_ALWAYS;
        sampler_info.minLod                  = 0.0f;
        sampler_info.maxLod                  = 8.0f; // todo: m_irradiance_texture_miplevels
        sampler_info.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sampler_info.unnormalizedCoordinates = VK_FALSE;

        if (vkCreateSampler(device, &sampler_info, nullptr, &m_linear_sampler) != VK_SUCCESS)
        {
            throw std::runtime_error("vk create sampler");
        }
    }

    return m_linear_sampler;
}

void Pilot::PVulkanUtil::destroyNearestSampler(VkDevice device)
{
    vkDestroySampler(device, m_nearest_sampler, nullptr);
    m_nearest_sampler = VK_NULL_HANDLE;
}

void Pilot::PVulkanUtil::destroyLinearSampler(VkDevice device)
{
    vkDestroySampler(device, m_linear_sampler, nullptr);
    m_linear_sampler = VK_NULL_HANDLE;
}

void Pilot::PVulkanUtil::bufferToImage(VkImage*              image,
                                       VkDeviceMemory*       memory,
                                       VkImageView*          view,
                                       class PVulkanContext* context,
                                       void*                 buffer,
                                       VkDeviceSize          buffer_size,
                                       VkFormat              format,
                                       uint32_t              width,
                                       uint32_t              height,
                                       VkImageUsageFlags     usage,
                                       VkImageLayout         layout)
{
    VkMemoryAllocateInfo mem_alloc_info = {};
    mem_alloc_info.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements mem_reqs;

    // Use a separate command buffer for texture loading
    VkCommandBuffer copy_cmd = context->beginSingleTimeCommands();

    // Create a host-visible staging buffer that contains the raw image data
    VkBuffer       staging_buffer;
    VkDeviceMemory staging_memory;

    VkBufferCreateInfo buffer_create_info = {};
    buffer_create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.size               = buffer_size;
    // This buffer is used as a transfer source for the buffer copy
    buffer_create_info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context->_device, &buffer_create_info, nullptr, &staging_buffer))
    {
        throw std::runtime_error("failed to create buffer");
    }

    // Get memory requirements for the staging buffer (alignment, memory type bits)
    vkGetBufferMemoryRequirements(context->_device, staging_buffer, &mem_reqs);

    mem_alloc_info.allocationSize = mem_reqs.size;
    // Get memory type index for a host visible buffer
    mem_alloc_info.memoryTypeIndex =
        findMemoryType(context->_physical_device,
                       mem_reqs.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(context->_device, &mem_alloc_info, nullptr, &staging_memory))
    {
        throw std::runtime_error("failed to allocate memory");
    }
    if (vkBindBufferMemory(context->_device, staging_buffer, staging_memory, 0))
    {
        throw std::runtime_error("failed to bind buffer");
    }

    // Copy texture data into staging buffer
    void* data;
    if (vkMapMemory(context->_device, staging_memory, 0, mem_reqs.size, 0, &data))
    {
        throw std::runtime_error("failed to map memory");
    }
    memcpy(data, buffer, buffer_size);
    vkUnmapMemory(context->_device, staging_memory);

    VkBufferImageCopy buffer_copy_region               = {};
    buffer_copy_region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    buffer_copy_region.imageSubresource.mipLevel       = 0;
    buffer_copy_region.imageSubresource.baseArrayLayer = 0;
    buffer_copy_region.imageSubresource.layerCount     = 1;
    buffer_copy_region.imageExtent.width               = width;
    buffer_copy_region.imageExtent.height              = height;
    buffer_copy_region.imageExtent.depth               = 1;
    buffer_copy_region.bufferOffset                    = 0;

    // Create optimal tiled target image
    VkImageCreateInfo image_create_info = {};
    image_create_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType         = VK_IMAGE_TYPE_2D;
    image_create_info.format            = format;
    image_create_info.mipLevels         = 1;
    image_create_info.arrayLayers       = 1;
    image_create_info.samples           = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.extent            = {width, height, 1};
    image_create_info.usage             = usage;
    // Ensure that the TRANSFER_DST bit is set for staging
    if (!(image_create_info.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    {
        image_create_info.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    if (vkCreateImage(context->_device, &image_create_info, nullptr, image))
    {
        throw std::runtime_error("failed to create image");
    }

    vkGetImageMemoryRequirements(context->_device, *image, &mem_reqs);

    mem_alloc_info.allocationSize = mem_reqs.size;

    mem_alloc_info.memoryTypeIndex =
        findMemoryType(context->_physical_device, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(context->_device, &mem_alloc_info, nullptr, memory))
    {
        throw std::runtime_error("failed to allocate memory");
    }
    if (vkBindImageMemory(context->_device, *image, *memory, 0))
    {
        throw std::runtime_error("failed to bind image memory");
    }

    VkImageSubresourceRange subresource_range = {};
    subresource_range.aspectMask              = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel            = 0;
    subresource_range.levelCount              = 1;
    subresource_range.layerCount              = 1;

    // Image barrier for optimal image (target)
    // Optimal image will be used as destination for the copy
    setImageLayout(
        copy_cmd, *image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource_range);

    // Copy mip levels from staging buffer
    vkCmdCopyBufferToImage(
        copy_cmd, staging_buffer, *image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &buffer_copy_region);

    // Change texture image layout to shader read after all mip levels have been copied
    setImageLayout(copy_cmd, *image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout, subresource_range);

    context->endSingleTimeCommands(copy_cmd);

    // Clean up staging resources
    vkFreeMemory(context->_device, staging_memory, nullptr);
    vkDestroyBuffer(context->_device, staging_buffer, nullptr);

    // Create image view
    VkImageViewCreateInfo view_create_info = {};
    view_create_info.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_create_info.pNext                 = nullptr;
    view_create_info.viewType              = VK_IMAGE_VIEW_TYPE_2D;
    view_create_info.format                = format;
    view_create_info.components            = {
                   VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    view_create_info.subresourceRange            = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    view_create_info.subresourceRange.levelCount = 1;
    view_create_info.image                       = *image;
    if (vkCreateImageView(context->_device, &view_create_info, nullptr, view))
    {
        throw std::runtime_error("failed to create image view");
    }
}

void Pilot::PVulkanUtil::setImageLayout(VkCommandBuffer         cmdbuffer,
                                        VkImage                 image,
                                        VkImageLayout           oldImageLayout,
                                        VkImageLayout           newImageLayout,
                                        VkImageSubresourceRange subresourceRange,
                                        VkPipelineStageFlags    srcStageMask,
                                        VkPipelineStageFlags    dstStageMask)
{
    // Create an image barrier object
    VkImageMemoryBarrier image_memory_barrier = {};
    image_memory_barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.oldLayout            = oldImageLayout;
    image_memory_barrier.newLayout            = newImageLayout;
    image_memory_barrier.image                = image;
    image_memory_barrier.subresourceRange     = subresourceRange;

    // Source layouts (old)
    // Source access mask controls actions that have to be finished on the old layout
    // before it will be transitioned to the new layout
    switch (oldImageLayout)
    {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            // Image layout is undefined (or does not matter)
            // Only valid as initial layout
            // No flags required, listed only for completeness
            image_memory_barrier.srcAccessMask = 0;
            break;

        case VK_IMAGE_LAYOUT_PREINITIALIZED:
            // Image is preinitialized
            // Only valid as initial layout for linear images, preserves memory contents
            // Make sure host writes have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            // Image is a color attachment
            // Make sure any writes to the color buffer have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            // Image is a depth/stencil attachment
            // Make sure any writes to the depth/stencil buffer have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            // Image is a transfer source
            // Make sure any reads from the image have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            // Image is a transfer destination
            // Make sure any writes to the image have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            // Image is read by a shader
            // Make sure any shader reads from the image have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            // Other source layouts aren't handled (yet)
            break;
    }

    // Target layouts (new)
    // Destination access mask controls the dependency for the new image layout
    switch (newImageLayout)
    {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            // Image will be used as a transfer destination
            // Make sure any writes to the image have been finished
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            // Image will be used as a transfer source
            // Make sure any reads from the image have been finished
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            // Image will be used as a color attachment
            // Make sure any writes to the color buffer have been finished
            image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            // Image layout will be used as a depth/stencil attachment
            // Make sure any writes to depth/stencil buffer have been finished
            image_memory_barrier.dstAccessMask =
                image_memory_barrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            // Image will be read in a shader (sampler, input attachment)
            // Make sure any writes to the image have been finished
            if (image_memory_barrier.srcAccessMask == 0)
            {
                image_memory_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            // Other source layouts aren't handled (yet)
            break;
    }

    // Put barrier inside setup command buffer
    vkCmdPipelineBarrier(cmdbuffer, srcStageMask, dstStageMask, 0, 0, nullptr, 0, nullptr, 1, &image_memory_barrier);
}