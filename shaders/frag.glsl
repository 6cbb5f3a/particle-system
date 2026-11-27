#version 460 core
in vec2 UV;

layout (location = 0) uniform sampler2D tex;

out vec4 fragcolor;
void main() {
    fragcolor = texture(tex, UV);
}