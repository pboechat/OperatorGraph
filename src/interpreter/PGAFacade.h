#pragma once

#include <memory>

#include <pga/core/GlobalConstants.h>
#include <pga/core/DispatchTable.h>
#include <pga/core/DynamicPolygon.cuh>
#include <pga/core/DynamicRightPrism.cuh>
#include <pga/compiler/ProcedureList.h>

//////////////////////////////////////////////////////////////////////////
typedef PGA::Shapes::DynamicPolygon<PGA::Constants::MaxNumSides, true> DCPoly;
typedef PGA::Shapes::DynamicPolygon<PGA::Constants::MaxNumSides, false> DPoly;
typedef PGA::Shapes::DynamicRightPrism<PGA::Constants::MaxNumSides, true> DCRPrism;
typedef PGA::Shapes::DynamicRightPrism<PGA::Constants::MaxNumSides, false> DRPrism;

//////////////////////////////////////////////////////////////////////////
// PGA control methods
PGA::Compiler::ProcedureList getProcedureList();
void initializePGA(const std::unique_ptr<PGA::DispatchTable>& dispatchTable);
double executePGA();
void releasePGA();
void destroyPGA();

