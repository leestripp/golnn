#version 450
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in mat4 aTransform;
layout(location = 6) in mat4 aPreviousTransform;
layout(set = 0, binding = 0) uniform SceneUniform {
    mat4 view;
    mat4 projection;
    float interpolation_alpha;
} scene;
layout(location = 0) out vec3 vertexColor;
void main()
{
    mat4 transform = aPreviousTransform * (1.0 - scene.interpolation_alpha)
        + aTransform * scene.interpolation_alpha;
    gl_Position = scene.projection * scene.view * transform * vec4(aPos, 1.0);
    vertexColor = aColor.rgb;
}
