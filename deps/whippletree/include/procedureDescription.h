#ifndef PROCEDUREDESCRIPTION_H_TXQCAGUB
#define PROCEDUREDESCRIPTION_H_TXQCAGUB

#include <string>

struct ProcedureDescription
{
  int ProcedureId;
  int NumThreads;
  bool ItemInput;
  int sharedMemory;
  bool InitialProcedure;
  int inputSize;
  std::string name;
  std::string algorithmname;

  template<class PROCEDURE, int proc_id>
  static ProcedureDescription generate()
  {
    ProcedureDescription d;
    d.ProcedureId = proc_id;
    d.NumThreads = PROCEDURE::NumThreads > 0 ? PROCEDURE::NumThreads : (PROCEDURE::ItemInput ? 1 : 0);
    d.ItemInput = PROCEDURE::ItemInput;
    d.sharedMemory = PROCEDURE::sharedMemory;
    d.InitialProcedure = PROCEDURE::InitialProcedure;
    d.inputSize = sizeof(typename PROCEDURE::ExpectedData);
    d.name = PROCEDURE::name();
    d.algorithmname = PROCEDURE::algorithmname();
    return d;
  }
};

#endif /* end of include guard: PROCEDUREDESCRIPTION_H_TXQCAGUB */
