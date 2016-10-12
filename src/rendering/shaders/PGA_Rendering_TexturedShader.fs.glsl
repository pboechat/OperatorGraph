#version 430

in vec3 f_normal;
in vec3 f_view;
in vec2 f_uv;

uniform sampler2D tex0;
uniform vec4 color0;
uniform float alphaThreshold = 1;
uniform float useLambertianReflectance = 1;

layout(location = 0) out vec4 color;

void main()
{
	vec3 n = normalize(f_normal);
	vec3 v = normalize(f_view);
	vec4 texel = texture(tex0, f_uv);
	if (texel.a < alphaThreshold)
		discard;
	float lambert = useLambertianReflectance * max(dot(n, v), 0) + float(useLambertianReflectance != 1);
	color = lambert * color0 * texel;
}