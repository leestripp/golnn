#version 450
layout(push_constant) uniform GridColor { vec4 color; } grid_color;
layout(location = 0) out vec4 FragColor;
void main() { FragColor = grid_color.color; }
