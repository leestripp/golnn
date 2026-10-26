#version 450
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(set = 0, binding = 0) uniform SceneUniform {
    mat4 view;
    mat4 projection;
    float interpolation_alpha;
} scene;
layout(location = 0) out vec3 vertexColor;
void main()
{
    gl_Position = scene.projection * scene.view * vec4(aPos, 1.0);
    gl_PointSize = 4.0;
    vertexColor = aColor;
}
