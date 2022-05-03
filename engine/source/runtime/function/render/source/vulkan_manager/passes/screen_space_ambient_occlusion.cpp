#include "runtime/function/render/include/render/vulkan_manager/vulkan_common.h"
#include "runtime/function/render/include/render/vulkan_manager/vulkan_mesh.h"
#include "runtime/function/render/include/render/vulkan_manager/vulkan_misc.h"
#include "runtime/function/render/include/render/vulkan_manager/vulkan_passes.h"
#include "runtime/function/render/include/render/vulkan_manager/vulkan_util.h"

#include "vulkan/vulkan_core.h"
#include <post_process_vert.h>
#include <screen_space_ambient_occlusion_frag.h>

namespace Pilot
{
    void PScreenSpaceAmbientOcclusionPass::initialize(VkRenderPass                    render_pass,
                                                      const std::vector<VkImageView>& input_attachments)
    {
        _framebuffer.render_pass = render_pass;
        setupDescriptorSetLayout();
        setupPipelines();
        setupDescriptorSet();
        updateAfterFramebufferRecreate(input_attachments);
    }

    void PScreenSpaceAmbientOcclusionPass::setupDescriptorSetLayout()
    {
        _descriptor_infos.resize(1);

        VkDescriptorSetLayoutBinding post_process_global_layout_bindings[5] = {};

        VkDescriptorSetLayoutBinding& gbuffer_a = post_process_global_layout_bindings[0];
        gbuffer_a.binding                       = 0;
        gbuffer_a.descriptorType                = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        gbuffer_a.descriptorCount               = 1;
        gbuffer_a.stageFlags                    = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutBinding& gbuffer_b = post_process_global_layout_bindings[1];
        gbuffer_b.binding                       = 1;
        gbuffer_b.descriptorType                = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        gbuffer_b.descriptorCount               = 1;
        gbuffer_b.stageFlags                    = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutBinding& gbuffer_c = post_process_global_layout_bindings[2];
        gbuffer_c.binding                       = 2;
        gbuffer_c.descriptorType                = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        gbuffer_c.descriptorCount               = 1;
        gbuffer_c.stageFlags                    = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutBinding& depth     = post_process_global_layout_bindings[3];
        depth.binding                           = 3;
        depth.descriptorType                    = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        depth.descriptorCount                   = 1;
        depth.stageFlags                        = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutBinding& deferred  = post_process_global_layout_bindings[4];
        deferred.binding                        = 4;
        deferred.descriptorType                 = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        deferred.descriptorCount                = 1;
        deferred.stageFlags                     = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo post_process_global_layout_create_info;
        post_process_global_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        post_process_global_layout_create_info.pNext = nullptr;
        post_process_global_layout_create_info.flags = 0;
        post_process_global_layout_create_info.bindingCount =
            sizeof(post_process_global_layout_bindings) / sizeof(post_process_global_layout_bindings[0]);
        post_process_global_layout_create_info.pBindings = post_process_global_layout_bindings;

        if (VK_SUCCESS != vkCreateDescriptorSetLayout(m_p_vulkan_context->_device,
                                                      &post_process_global_layout_create_info,
                                                      nullptr,
                                                      &_descriptor_infos[0].layout))
        {
            throw std::runtime_error("create post process global layout");
        }
    }

    void PScreenSpaceAmbientOcclusionPass::setupPipelines()
    {
        _render_pipelines.resize(1);

        VkDescriptorSetLayout      descriptorset_layouts[1] = {_descriptor_infos[0].layout};
        VkPipelineLayoutCreateInfo pipeline_layout_create_info {};
        pipeline_layout_create_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_create_info.setLayoutCount = 1;
        pipeline_layout_create_info.pSetLayouts    = descriptorset_layouts;

        if (vkCreatePipelineLayout(
                m_p_vulkan_context->_device, &pipeline_layout_create_info, nullptr, &_render_pipelines[0].layout) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("create post process pipeline layout");
        }

        VkShaderModule vert_shader_module =
            PVulkanUtil::createShaderModule(m_p_vulkan_context->_device, POST_PROCESS_VERT);
        VkShaderModule frag_shader_module =
            PVulkanUtil::createShaderModule(m_p_vulkan_context->_device, SCREEN_SPACE_AMBIENT_OCCLUSION_FRAG);

        VkPipelineShaderStageCreateInfo vert_pipeline_shader_stage_create_info {};
        vert_pipeline_shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert_pipeline_shader_stage_create_info.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vert_pipeline_shader_stage_create_info.module = vert_shader_module;
        vert_pipeline_shader_stage_create_info.pName  = "main";

        VkPipelineShaderStageCreateInfo frag_pipeline_shader_stage_create_info {};
        frag_pipeline_shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_pipeline_shader_stage_create_info.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_pipeline_shader_stage_create_info.module = frag_shader_module;
        frag_pipeline_shader_stage_create_info.pName  = "main";

        VkPipelineShaderStageCreateInfo shader_stages[] = {vert_pipeline_shader_stage_create_info,
                                                           frag_pipeline_shader_stage_create_info};

        VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info {};
        vertex_input_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_state_create_info.vertexBindingDescriptionCount   = 0;
        vertex_input_state_create_info.pVertexBindingDescriptions      = nullptr;
        vertex_input_state_create_info.vertexAttributeDescriptionCount = 0;
        vertex_input_state_create_info.pVertexAttributeDescriptions    = nullptr;

        VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info {};
        input_assembly_create_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly_create_info.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        input_assembly_create_info.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewport_state_create_info {};
        viewport_state_create_info.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state_create_info.viewportCount = 1;
        viewport_state_create_info.pViewports    = &m_command_info._viewport;
        viewport_state_create_info.scissorCount  = 1;
        viewport_state_create_info.pScissors     = &m_command_info._scissor;

        VkPipelineRasterizationStateCreateInfo rasterization_state_create_info {};
        rasterization_state_create_info.sType            = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization_state_create_info.depthClampEnable = VK_FALSE;
        rasterization_state_create_info.rasterizerDiscardEnable = VK_FALSE;
        rasterization_state_create_info.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterization_state_create_info.lineWidth               = 1.0f;
        rasterization_state_create_info.cullMode                = VK_CULL_MODE_BACK_BIT;
        rasterization_state_create_info.frontFace               = VK_FRONT_FACE_CLOCKWISE;
        rasterization_state_create_info.depthBiasEnable         = VK_FALSE;
        rasterization_state_create_info.depthBiasConstantFactor = 0.0f;
        rasterization_state_create_info.depthBiasClamp          = 0.0f;
        rasterization_state_create_info.depthBiasSlopeFactor    = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisample_state_create_info {};
        multisample_state_create_info.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample_state_create_info.sampleShadingEnable  = VK_FALSE;
        multisample_state_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState color_blend_attachment_state {};
        color_blend_attachment_state.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment_state.blendEnable         = VK_FALSE;
        color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachment_state.colorBlendOp        = VK_BLEND_OP_ADD;
        color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachment_state.alphaBlendOp        = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo color_blend_state_create_info {};
        color_blend_state_create_info.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend_state_create_info.logicOpEnable     = VK_FALSE;
        color_blend_state_create_info.logicOp           = VK_LOGIC_OP_COPY;
        color_blend_state_create_info.attachmentCount   = 1;
        color_blend_state_create_info.pAttachments      = &color_blend_attachment_state;
        color_blend_state_create_info.blendConstants[0] = 0.0f;
        color_blend_state_create_info.blendConstants[1] = 0.0f;
        color_blend_state_create_info.blendConstants[2] = 0.0f;
        color_blend_state_create_info.blendConstants[3] = 0.0f;

        VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info {};
        depth_stencil_create_info.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil_create_info.depthTestEnable       = VK_TRUE;
        depth_stencil_create_info.depthWriteEnable      = VK_TRUE;
        depth_stencil_create_info.depthCompareOp        = VK_COMPARE_OP_LESS;
        depth_stencil_create_info.depthBoundsTestEnable = VK_FALSE;
        depth_stencil_create_info.stencilTestEnable     = VK_FALSE;

        VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

        VkPipelineDynamicStateCreateInfo dynamic_state_create_info {};
        dynamic_state_create_info.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state_create_info.dynamicStateCount = 2;
        dynamic_state_create_info.pDynamicStates    = dynamic_states;

        VkGraphicsPipelineCreateInfo pipeline_info {};
        pipeline_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount          = 2;
        pipeline_info.pStages             = shader_stages;
        pipeline_info.pVertexInputState   = &vertex_input_state_create_info;
        pipeline_info.pInputAssemblyState = &input_assembly_create_info;
        pipeline_info.pViewportState      = &viewport_state_create_info;
        pipeline_info.pRasterizationState = &rasterization_state_create_info;
        pipeline_info.pMultisampleState   = &multisample_state_create_info;
        pipeline_info.pColorBlendState    = &color_blend_state_create_info;
        pipeline_info.pDepthStencilState  = &depth_stencil_create_info;
        pipeline_info.layout              = _render_pipelines[0].layout;
        pipeline_info.renderPass          = _framebuffer.render_pass;
        pipeline_info.subpass             = _custom_screen_space_ambient_occlusion;
        pipeline_info.basePipelineHandle  = VK_NULL_HANDLE;
        pipeline_info.pDynamicState       = &dynamic_state_create_info;

        if (vkCreateGraphicsPipelines(m_p_vulkan_context->_device,
                                      VK_NULL_HANDLE,
                                      1,
                                      &pipeline_info,
                                      nullptr,
                                      &_render_pipelines[0].pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("create post process graphics pipeline");
        }

        vkDestroyShaderModule(m_p_vulkan_context->_device, vert_shader_module, nullptr);
        vkDestroyShaderModule(m_p_vulkan_context->_device, frag_shader_module, nullptr);
    }

    void PScreenSpaceAmbientOcclusionPass::setupDescriptorSet()
    {
        VkDescriptorSetAllocateInfo post_process_global_descriptor_set_alloc_info;
        post_process_global_descriptor_set_alloc_info.sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        post_process_global_descriptor_set_alloc_info.pNext          = nullptr;
        post_process_global_descriptor_set_alloc_info.descriptorPool = m_descriptor_pool;
        post_process_global_descriptor_set_alloc_info.descriptorSetCount = 1;
        post_process_global_descriptor_set_alloc_info.pSetLayouts        = &_descriptor_infos[0].layout;

        if (VK_SUCCESS != vkAllocateDescriptorSets(m_p_vulkan_context->_device,
                                                   &post_process_global_descriptor_set_alloc_info,
                                                   &_descriptor_infos[0].descriptor_set))
        {
            throw std::runtime_error("allocate post process global descriptor set");
        }
    }

    void
    PScreenSpaceAmbientOcclusionPass::updateAfterFramebufferRecreate(const std::vector<VkImageView>& input_attachments)
    {
        VkDescriptorImageInfo post_process_per_frame_gbuffer_a_attachment_info = {};
        post_process_per_frame_gbuffer_a_attachment_info.sampler =
            PVulkanUtil::getOrCreateNearestSampler(m_p_vulkan_context->_physical_device, m_p_vulkan_context->_device);
        post_process_per_frame_gbuffer_a_attachment_info.imageView   = input_attachments[_main_camera_pass_gbuffer_a];
        post_process_per_frame_gbuffer_a_attachment_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo post_process_per_frame_gbuffer_b_attachment_info = {};
        post_process_per_frame_gbuffer_b_attachment_info.sampler =
            PVulkanUtil::getOrCreateNearestSampler(m_p_vulkan_context->_physical_device, m_p_vulkan_context->_device);
        post_process_per_frame_gbuffer_b_attachment_info.imageView   = input_attachments[_main_camera_pass_gbuffer_b];
        post_process_per_frame_gbuffer_b_attachment_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo post_process_per_frame_gbuffer_c_attachment_info = {};
        post_process_per_frame_gbuffer_c_attachment_info.sampler =
            PVulkanUtil::getOrCreateNearestSampler(m_p_vulkan_context->_physical_device, m_p_vulkan_context->_device);
        post_process_per_frame_gbuffer_c_attachment_info.imageView   = input_attachments[_main_camera_pass_gbuffer_c];
        post_process_per_frame_gbuffer_c_attachment_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo depth_input_attachment_info = {};
        depth_input_attachment_info.sampler =
            PVulkanUtil::getOrCreateNearestSampler(m_p_vulkan_context->_physical_device, m_p_vulkan_context->_device);
        depth_input_attachment_info.imageView   = m_p_vulkan_context->_depth_image_view;
        depth_input_attachment_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo color_attachment_info = {};
        color_attachment_info.sampler =
            PVulkanUtil::getOrCreateNearestSampler(m_p_vulkan_context->_physical_device, m_p_vulkan_context->_device);
        color_attachment_info.imageView   = input_attachments[_main_camera_pass_backup_buffer_odd];
        color_attachment_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet post_process_descriptor_writes_info[5];

        VkWriteDescriptorSet& post_process_descriptor_gbuffer_a_attachment_write_info =
            post_process_descriptor_writes_info[0];
        post_process_descriptor_gbuffer_a_attachment_write_info.sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        post_process_descriptor_gbuffer_a_attachment_write_info.pNext      = nullptr;
        post_process_descriptor_gbuffer_a_attachment_write_info.dstSet     = _descriptor_infos[0].descriptor_set;
        post_process_descriptor_gbuffer_a_attachment_write_info.dstBinding = 0;
        post_process_descriptor_gbuffer_a_attachment_write_info.dstArrayElement = 0;
        post_process_descriptor_gbuffer_a_attachment_write_info.descriptorType  = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        post_process_descriptor_gbuffer_a_attachment_write_info.descriptorCount = 1;
        post_process_descriptor_gbuffer_a_attachment_write_info.pImageInfo =
            &post_process_per_frame_gbuffer_a_attachment_info;

        VkWriteDescriptorSet& post_process_descriptor_gbuffer_b_attachment_write_info =
            post_process_descriptor_writes_info[1];
        post_process_descriptor_gbuffer_b_attachment_write_info.sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        post_process_descriptor_gbuffer_b_attachment_write_info.pNext      = nullptr;
        post_process_descriptor_gbuffer_b_attachment_write_info.dstSet     = _descriptor_infos[0].descriptor_set;
        post_process_descriptor_gbuffer_b_attachment_write_info.dstBinding = 1;
        post_process_descriptor_gbuffer_b_attachment_write_info.dstArrayElement = 0;
        post_process_descriptor_gbuffer_b_attachment_write_info.descriptorType  = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        post_process_descriptor_gbuffer_b_attachment_write_info.descriptorCount = 1;
        post_process_descriptor_gbuffer_b_attachment_write_info.pImageInfo =
            &post_process_per_frame_gbuffer_b_attachment_info;

        VkWriteDescriptorSet& post_process_descriptor_gbuffer_c_attachment_write_info =
            post_process_descriptor_writes_info[2];
        post_process_descriptor_gbuffer_c_attachment_write_info.sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        post_process_descriptor_gbuffer_c_attachment_write_info.pNext      = nullptr;
        post_process_descriptor_gbuffer_c_attachment_write_info.dstSet     = _descriptor_infos[0].descriptor_set;
        post_process_descriptor_gbuffer_c_attachment_write_info.dstBinding = 2;
        post_process_descriptor_gbuffer_c_attachment_write_info.dstArrayElement = 0;
        post_process_descriptor_gbuffer_c_attachment_write_info.descriptorType  = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        post_process_descriptor_gbuffer_c_attachment_write_info.descriptorCount = 1;
        post_process_descriptor_gbuffer_c_attachment_write_info.pImageInfo =
            &post_process_per_frame_gbuffer_c_attachment_info;

        VkWriteDescriptorSet& depth_descriptor_input_attachment_write_info = post_process_descriptor_writes_info[3];
        depth_descriptor_input_attachment_write_info.sType                 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        depth_descriptor_input_attachment_write_info.pNext                 = nullptr;
        depth_descriptor_input_attachment_write_info.dstSet                = _descriptor_infos[0].descriptor_set;
        depth_descriptor_input_attachment_write_info.dstBinding            = 3;
        depth_descriptor_input_attachment_write_info.dstArrayElement       = 0;
        depth_descriptor_input_attachment_write_info.descriptorType        = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        depth_descriptor_input_attachment_write_info.descriptorCount       = 1;
        depth_descriptor_input_attachment_write_info.pImageInfo            = &depth_input_attachment_info;

        VkWriteDescriptorSet& color_input_attachment_write_info = post_process_descriptor_writes_info[4];
        color_input_attachment_write_info.sType                 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        color_input_attachment_write_info.pNext                 = nullptr;
        color_input_attachment_write_info.dstSet                = _descriptor_infos[0].descriptor_set;
        color_input_attachment_write_info.dstBinding            = 4;
        color_input_attachment_write_info.dstArrayElement       = 0;
        color_input_attachment_write_info.descriptorType        = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        color_input_attachment_write_info.descriptorCount       = 1;
        color_input_attachment_write_info.pImageInfo            = &color_attachment_info;

        vkUpdateDescriptorSets(m_p_vulkan_context->_device,
                               sizeof(post_process_descriptor_writes_info) /
                                   sizeof(post_process_descriptor_writes_info[0]),
                               post_process_descriptor_writes_info,
                               0,
                               nullptr);
    }

    void PScreenSpaceAmbientOcclusionPass::draw()
    {
        if (m_render_config._enable_debug_untils_label)
        {
            VkDebugUtilsLabelEXT label_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                                               nullptr,
                                               "Screen Space Ambient Occlusion",
                                               {1.0f, 1.0f, 1.0f, 1.0f}};
            m_p_vulkan_context->_vkCmdBeginDebugUtilsLabelEXT(m_command_info._current_command_buffer, &label_info);
        }

        m_p_vulkan_context->_vkCmdBindPipeline(
            m_command_info._current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _render_pipelines[0].pipeline);
        m_p_vulkan_context->_vkCmdSetViewport(m_command_info._current_command_buffer, 0, 1, &m_command_info._viewport);
        m_p_vulkan_context->_vkCmdSetScissor(m_command_info._current_command_buffer, 0, 1, &m_command_info._scissor);
        m_p_vulkan_context->_vkCmdBindDescriptorSets(m_command_info._current_command_buffer,
                                                     VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                     _render_pipelines[0].layout,
                                                     0,
                                                     1,
                                                     &_descriptor_infos[0].descriptor_set,
                                                     0,
                                                     nullptr);

        vkCmdDraw(m_command_info._current_command_buffer, 3, 1, 0, 0);

        if (m_render_config._enable_debug_untils_label)
        {
            m_p_vulkan_context->_vkCmdEndDebugUtilsLabelEXT(m_command_info._current_command_buffer);
        }
    }

} // namespace Pilot
