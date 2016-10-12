#version 430

#include <attributes.glsl>
#include <camera.glsl>

layout(location = 0) in VertexAttributes v_vertex;
layout(location = 10) in InstanceAttributes v_instance;

out vec3 f_normal;
out vec2 f_uv;
out vec3 f_view;

void main()
{
	mat4 model = v_instance.modelMatrix;
	vec4 world = model * vec4(v_vertex.position, 1);
	f_normal = (model * vec4(v_vertex.normal, 0)).xyz;
	f_uv = v_vertex.uv;
	f_view = camera.Position - world.xyz;
	gl_Position = camera.ViewProjection * world;
}
