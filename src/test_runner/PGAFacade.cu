#define PGA_CORE_EXPORT 0
#define PGA_RENDERING_EXPORT 0

#include <string.h>
#include <stdexcept>

// NOTE: DebugFlags.h have to come before Core.h because of partial template specializations!
#include "DebugFlags.h"

#include <pga/core/Core.h>
#include <pga/core/GPUTechnique.h>
#include <pga/rendering/GenerationFunctions.cuh>

#include "SceneUtils.cuh"
#include "PGAFacade.h"

//////////////////////////////////////////////////////////////////////////
#include "SceneSelector.cuh"

namespace Test
{
	using namespace Scene;

}

//////////////////////////////////////////////////////////////////////////
unsigned int g_attributeIndex = 0;

std::string getTestName()
{
	return Test::testName();
}

std::string getSceneName()
{
	return Test::sceneName();
}

std::string getConfigurationString()
{
	return SceneUtils::SceneConfiguration<Test::Configurator>::getConfigurationString();
}

unsigned int getNumAxioms()
{
	return Test::AxiomGenerator::getNumAxioms();
}

void incrementNumElements()
{
	SceneUtils::SceneGrid<Test::Controller>::incrementNumElements();
}

void decrementNumElements()
{
	SceneUtils::SceneGrid<Test::Controller>::decrementNumElements();
}

void incrementGridSize()
{
	SceneUtils::SceneGrid<Test::Controller>::incrementGridSize();
}

void decrementGridSize()
{
	SceneUtils::SceneGrid<Test::Controller>::decrementGridSize();
}

void incrementAttribute()
{
	SceneUtils::SceneAttributes<Test::Controller>::increment(g_attributeIndex);
}

void decrementAttribute()
{
	SceneUtils::SceneAttributes<Test::Controller>::decrement(g_attributeIndex);
}

unsigned int getNumElements()
{
	return SceneUtils::SceneGrid<Test::Controller>::getNumElements();
}

void setNumElements(unsigned int numElements)
{
	SceneUtils::SceneGrid<Test::Controller>::setNumElements(numElements);
}

void setAttributeIndex(unsigned int attributeIndex)
{
	g_attributeIndex = attributeIndex;
}

void setAttributeValue(unsigned int value)
{
	SceneUtils::SceneAttributes<Test::Controller>::setAttribute(g_attributeIndex, value);
}

void maximizeNumElements()
{
	SceneUtils::SceneGrid<Test::Controller>::maximizeNumElements();
}

bool isInstrumented()
{
	return Test::Instrumented;
}

//////////////////////////////////////////////////////////////////////////
struct Configuration
{
#if defined(PGA_CPU)
	static const unsigned int MaxDerivationSteps = 100000;
#else
	static const PGA::GPU::Technique Technique = Test::Technique;
	static const unsigned int QueueSize = Test::QueueSize;
	static const unsigned int MaxSharedMemory = 0;
#endif

};

typedef PGA::SinglePhaseEvaluator<Test::ProcedureList, Test::AxiomGenerator, PGA::Rendering::GenFuncFilter, Test::Instrumented, Test::NumSubgraphs, Test::NumEdges, Configuration> Evaluator;
typedef std::unique_ptr<Evaluator, PGA::ReleaseCallback> EvaluatorPtr;
EvaluatorPtr g_evaluator;

//////////////////////////////////////////////////////////////////////////
void initializePGA()
{
	g_evaluator = EvaluatorPtr(new Evaluator());
	g_evaluator->initialize(Test::dispatchTable.toDispatchTableEntriesPtr().get(), Test::dispatchTable.entries.size());
	SceneUtils::SceneInitialization<Test::Controller>::initialize();
}

double executePGA()
{
	return g_evaluator->execute(Test::AxiomGenerator::getNumAxioms());
}

void destroyPGA()
{
	g_evaluator = 0;
}

void releasePGA()
{
	g_evaluator.release();
}