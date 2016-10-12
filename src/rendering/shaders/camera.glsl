struct Camera
{
	mat4x4 View;
	mat4x4 Projection;
	mat4x4 ViewProjection;
	vec3 Position;

};

layout(std140, row_major) uniform CameraParameters
{
	Camera camera;
  
};


