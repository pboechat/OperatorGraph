#version 430

#include <attributes.glsl>
#include <camera.glsl>

layout(location = 0) in VertexAttributes v_vertex;
layout(location = 10) in InstanceAttributes v_instance;

uniform float useUvX0 = 1.0f;
uniform float useUvY0 = 1.0f;
uniform vec2 uvScale0 = vec2(1.0f, 1.0f);

out vec3 f_normal;
out vec3 f_view;
out vec2 f_uv;

void main()
{
	mat4 model = v_instance.modelMatrix;
	vec4 world = model * vec4(v_vertex.position, 1);
	f_normal = (model * vec4(v_vertex.normal, 0)).xyz;
	f_view = camera.Position - world.xyz;
	f_uv = v_vertex.uv;
	vec3 abs_normal = abs(f_normal);
	if (abs_normal.x > abs_normal.y)
	{
		if (abs_normal.x > abs_normal.z)
			f_uv = vec2(mix(world.z, f_uv.x, useUvX0), mix(world.y, f_uv.y, useUvY0));
		else
			f_uv = vec2(mix(world.x, f_uv.x, useUvX0), mix(world.y, f_uv.y, useUvY0));
	}
	else
	{
		if (abs_normal.y > abs_normal.z)
			f_uv = vec2(mix(world.x, f_uv.x, useUvX0), mix(world.z, f_uv.y, useUvY0));
		else
			f_uv = vec2(mix(world.x, f_uv.x, useUvX0), mix(world.y, f_uv.y, useUvY0));
	}
	f_uv *= uvScale0;
	gl_Position = camera.ViewProjection * world;
}
