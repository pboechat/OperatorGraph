#include <camera.hlsli>

//////////////
// TYPEDEFS //
//////////////
struct VSInput
{
	float4 position: POSITION;
	float3 normal: NORMAL;
	float2 uv: TEXCOORD0;
	float custom0 : CUSTOM0;
	float4x4 modelMatrix: MODELMATRIX;
	float custom1 : CUSTOM1;

};

struct PSInput
{
	float4 position: SV_POSITION;
	float3 normal: NORMAL;
	float2 uv: TEXCOORD0;
	float3 viewDirection: TEXCOORD1;

};

PSInput main(VSInput i)
{
	PSInput o;
	float4 world = mul(i.position, i.modelMatrix);
	o.position = mul(world, camera.ViewProjection);
	o.normal = mul(float4(i.normal, 0.0f), i.modelMatrix).xyz;
	o.uv = i.uv;
	o.viewDirection = camera.Position - world.xyz;
	return o;
}