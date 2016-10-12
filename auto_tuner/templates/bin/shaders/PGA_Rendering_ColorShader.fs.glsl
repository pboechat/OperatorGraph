#version 430

in vec3 f_normal;
in vec2 f_uv;
in vec3 f_view;

uniform vec4 color0 = vec4(1, 1, 1, 1);

layout(location = 0) out vec4 color;

void main()
{
	vec3 n = normalize(f_normal);
	vec3 v = normalize(f_view);
	float lambert = max(dot(n, v), 0);
	vec2 d = abs(f_uv - 0.5f) * 2;
	color = color0 * mix(1, 0.8f, pow(length(d), 3)) * vec4(lambert);
}
