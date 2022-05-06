#version 310 es

#extension GL_GOOGLE_include_directive : enable

#include "constants.h"

layout(location = 0) out vec2 out_uv;

void main()
{
    const vec2 fullscreen_triangle_positions[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position                                 = vec4(fullscreen_triangle_positions[gl_VertexIndex], 0, 1);
    out_uv                                      = gl_Position.xy * 0.5 + 0.5;
}
