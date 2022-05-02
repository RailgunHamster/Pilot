#version 310 es

#extension GL_GOOGLE_include_directive : enable

#include "constants.h"

layout(input_attachment_index = 0, set = 0, binding = 0) uniform highp subpassInput in_color;

layout(set = 0, binding = 1) uniform sampler2D color_grading_lut_texture_sampler;

layout(location = 0) in highp vec2 in_uv;
layout(location = 0) out highp vec4 out_color;

void main()
{
    highp ivec2 lut_tex_size = textureSize(color_grading_lut_texture_sampler, 0);
    highp float size         = float(lut_tex_size.y);
    highp vec4  color        = subpassLoad(in_color).rgba;
    highp float int_b        = floor(color.b * size);
    highp float frac_b       = color.b * size - int_b;
    highp float u            = (color.r + int_b) / size;
    highp float v            = color.g;
    highp vec3  RG0          = texture(color_grading_lut_texture_sampler, vec2(u, v)).rgb;
    highp vec3  RG1          = texture(color_grading_lut_texture_sampler, vec2(u + 1.0 / size, v)).rgb;
    highp vec3  result       = RG0 * (1.0 - frac_b) + RG1 * frac_b;
    out_color                = vec4(result, 1.0);
}
