#pragma once

#include <string.h>
#include <vector>
#include <cuda_runtime_api.h>

#include <math/vector.h>
#include <pga/core/CUDAException.h>

#include "Constants.h"
#include "Axiom.h"

#if defined(INTERPRETER_EXTERN)
namespace Host
{
	extern Axiom* setAxioms(unsigned int numAxioms, Axiom* axioms);

}

namespace Device
{
	extern Axiom* setAxioms(unsigned int numAxioms, Axiom* axioms);

}
#else
namespace Host
{
	unsigned int numAxioms;
	Axiom* axioms;

	Axiom* setAxioms(unsigned int numAxioms, Axiom* axioms)
	{
		Host::numAxioms = numAxioms;
		size_t size = numAxioms * sizeof(Axiom);
		Host::axioms = (Axiom*)malloc(size);
		memcpy(Host::axioms, axioms, size);
		return Host::axioms;
	}

}

namespace Device
{
	__device__ __constant__ unsigned int numAxioms;
	__device__ Axiom* axioms;

	Axiom* setAxioms(unsigned int numAxioms, Axiom* axioms)
	{
		::Host::numAxioms = numAxioms;
		size_t size = numAxioms * sizeof(Axiom);
		::Host::axioms = (Axiom*)malloc(size);
		memcpy(::Host::axioms, axioms, size);
		Axiom* dAxioms;
		PGA_CUDA_checkedCall(cudaMemcpyToSymbol(::Device::numAxioms, &numAxioms, sizeof(unsigned int)));
		PGA_CUDA_checkedCall(cudaMalloc((void **)&dAxioms, size));
		PGA_CUDA_checkedCall(cudaMemcpy(dAxioms, axioms, size, cudaMemcpyHostToDevice));
		PGA_CUDA_checkedCall(cudaMemcpyToSymbol(::Device::axioms, &dAxioms, sizeof(Axiom*)));
		return dAxioms;
	}

}
#endif
