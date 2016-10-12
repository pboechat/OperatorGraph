struct DLight
{
	float4 color;
	float4 direction;

};

cbuffer LightingParameters : register(b1)
{
	DLight dLight;

};

float lambert(float4 lightDirection, float3 normal)
{
	return max(0, dot(normal, lightDirection.xyz));
}
