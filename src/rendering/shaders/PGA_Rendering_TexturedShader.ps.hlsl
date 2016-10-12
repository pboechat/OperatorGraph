#include <lighting.hlsli>

//////////////
// TYPEDEFS //
//////////////
struct PSInput
{
	float4 position: SV_POSITION;
	float3 normal: NORMAL;
	float2 uv: TEXCOORD0;
	float3 viewDirection: TEXCOORD1;

};

/////////////
// GLOBALS //
/////////////
cbuffer MaterialParameters : register(b2)
{
	float alphaThreshold;
	float useColorLookup;
	float useUvX0;
	float useUvY0;
	float2 uvScale0;
	float4 color0;

};

Texture2D tex0 : register(t0);
SamplerState sampler0 : register (s0);

float4 main(PSInput i) : SV_TARGET
{
	float4 texel = tex0.Sample(sampler0, i.uv);
	if (texel.a < alphaThreshold)
		discard;
	return lambert(dLight.direction, i.normal) * dLight.color * color0 * texel;
}