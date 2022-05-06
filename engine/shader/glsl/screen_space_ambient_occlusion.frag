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
layout(binding = 8) uniform in_projection
{
    highp mat4  projection;
    highp mat4  view;
    highp vec4  viewport;
    highp vec2  extent;
    highp float znear;
    highp float zfar;
    highp int   state;
};

layout(location = 0) in highp vec2 in_uv;
layout(location = 0) out highp vec4 out_color;

highp vec2 toUV(highp vec2 src)
{
    highp float u = (src.x * viewport.z + viewport.x) / extent.x;
    highp float v = (src.y * viewport.w + viewport.y) / extent.y;
    return vec2(u, v);
}

highp float linearDepth(highp float depth)
{
    highp float z = depth * 2.0 - 1.0;
    return -(2.0 * znear * zfar) / (zfar + znear - z * (zfar - znear));
}

void main()
{
    // uv
    highp vec2 uv = toUV(in_uv);

    // pos & normal
    highp vec3 view_pos      = vec3(view * texture(in_gbuffer_d, uv).rgba);
    highp mat3 normal_matrix = transpose(inverse(mat3(view)));
    highp vec3 view_normal   = DecodeNormal(texture(in_gbuffer_a, uv).rgb);
    view_normal              = normal_matrix * view_normal;
    view_normal              = normalize(view_normal);
    highp float depth        = linearDepth(texture(in_scene_depth, uv).r);

    // noise vector
    highp ivec2 tex_dim    = textureSize(in_gbuffer_d, 0);
    highp ivec2 noise_dim  = textureSize(in_noise, 0);
    highp vec2  noise_uv   = vec2(float(tex_dim.x) / float(noise_dim.x), float(tex_dim.y) / float(noise_dim.y)) * uv;
    highp vec3  random_vec = texture(in_noise, noise_uv).xyz;

    // TBN
    highp vec3 tangent   = normalize(random_vec - view_normal * dot(random_vec, view_normal));
    highp vec3 bitangent = cross(tangent, view_normal);
    highp mat3 TBN       = mat3(tangent, bitangent, view_normal);

    // occlusion
    highp float occlusion = 0.0;
    highp float bias      = 0.025;
    for (int i = 0; i < SSAO_KERNEL_SIZE; ++i)
    {
        highp vec3 sample_pos = TBN * samples[i].xyz * SSAO_RADIUS + view_pos;
        // projection
        highp vec4 offset = vec4(sample_pos, 1.0);
        offset            = projection * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        highp float sample_depth = linearDepth(texture(in_scene_depth, toUV(offset.xy)).r);
        highp float check        = smoothstep(0.0, 1.0, SSAO_RADIUS / abs(depth - sample_depth));
        occlusion += (sample_depth >= sample_pos.z + bias ? 1.0 : 0.0) * check;
    }
    occlusion = 1.0 - (occlusion / float(SSAO_KERNEL_SIZE));

    highp vec4 color = subpassLoad(in_color).rgba;
    if (state == 0)
    {
        out_color = vec4(color.rgb * occlusion, color.a);
    }
    else if (state == 1)
    {
        out_color = color;
    }
    else if (state == 2)
    {
        out_color = vec4(vec3(occlusion), 1.0);
    }
}
