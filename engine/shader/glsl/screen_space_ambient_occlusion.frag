#version 310 es

#extension GL_GOOGLE_include_directive : enable

#include "constants.h"
#include "gbuffer.h"

layout(constant_id = 0) const highp int SSAO_KERNEL_SIZE = 64;
layout(constant_id = 1) const highp float SSAO_RADIUS    = 0.5;

layout(binding = 0) uniform highp sampler2D in_gbuffer_a;
layout(binding = 1) uniform highp sampler2D in_gbuffer_b;
layout(binding = 2) uniform highp sampler2D in_gbuffer_c;
layout(binding = 3) uniform highp sampler2D in_gbuffer_d;
layout(binding = 4) uniform highp sampler2D in_scene_depth;
layout(input_attachment_index = 0, binding = 5) uniform highp subpassInput in_color;
layout(binding = 6) uniform highp sampler2D in_noise;
layout(binding = 7) uniform in_samples { highp vec4 samples[SSAO_KERNEL_SIZE]; };

layout(location = 0) in highp vec2 in_uv;
layout(location = 0) out highp vec4 out_color;

void main()
{
    PGBufferData gbuffer;
    highp vec4   gbuffer_a = texture(in_gbuffer_a, in_uv).rgba;
    highp vec4   gbuffer_b = texture(in_gbuffer_b, in_uv).rgba;
    highp vec4   gbuffer_c = texture(in_gbuffer_c, in_uv).rgba;
    highp vec4   gbuffer_d = texture(in_gbuffer_d, in_uv).rgba;
    DecodeGBufferData(gbuffer, gbuffer_a, gbuffer_b, gbuffer_c, gbuffer_d);
    highp float depth = texture(in_scene_depth, in_uv).r;
    highp vec3  noise = texture(in_noise, in_uv).rgb;
    highp vec4  s     = samples[0];

    out_color = subpassLoad(in_color).rgba;
}
