#version 310 es

#extension GL_GOOGLE_include_directive : enable

#include "constants.h"
#include "gbuffer.h"

layout(input_attachment_index = 0, set = 0, binding = 0) uniform highp subpassInput in_gbuffer_a;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform highp subpassInput in_gbuffer_b;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform highp subpassInput in_gbuffer_c;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform highp subpassInput in_scene_depth;
layout(input_attachment_index = 4, set = 0, binding = 4) uniform highp subpassInput in_color;

layout(location = 0) in highp vec2 in_uv;
layout(location = 0) out highp vec4 out_color;

void main()
{
    out_color = subpassLoad(in_gbuffer_a).rgba;
    out_color = subpassLoad(in_gbuffer_b).rgba;
    out_color = subpassLoad(in_gbuffer_c).rgba;
    out_color = subpassLoad(in_scene_depth).rgba;
    out_color = subpassLoad(in_color).rgba;
}
