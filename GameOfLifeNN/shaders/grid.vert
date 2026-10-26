#version 450
layout(location = 0) in vec3 aPos;
layout(set = 0, binding = 0) uniform SceneUniform {
    mat4 view;
    mat4 projection;
    float interpolation_alpha;
} scene;
void main() { gl_Position = scene.projection * scene.view * vec4(aPos, 1.0); }
