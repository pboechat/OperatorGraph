//#pragma once
//#include "queueInterface.cuh"
//#include "common.cuh"
//#include "random.cuh"
//
//namespace Distributed
//{
//  template<class ProcedureInfo, class QueueElement = void>
//  class NoQueue
//  {
//  public:
//    __inline__ __device__ void init() 
//    { }
//    template<class PROCEDURE>
//    __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData const& data) 
//    { return false;  }
//    template<class PROCEDURE>
//    __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData const& data) 
//    {  return false; }
//
//    __inline__ __device__ int dequeue(QueueElement* data, int& procId, int startThread = 0)
//    {  return 0; }
//
//    __inline__ __device__ int size() const
//    {  return 0; }
//
//    static std::string name()
//    {  return ""; }
//  };
//
//  template<class PROCEDURE, class ProcedureInfo, template<class ProcedureInfo, class QueueElement> class InternalPackageQueue, template<class ProcedureInfo, class QueueElement> class InternalItemQueue, bool Itemized, int procIdSel >
//  struct QueueSelector
//  {
//  };
//  template<class PROCEDURE, class ProcedureInfo, template<class ProcedureInfo, class QueueElement> class InternalPackageQueue, template<class ProcedureInfo, class QueueElement> class InternalItemQueue, int procIdSel>
//  struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, InternalItemQueue, true, procIdSel> : public InternalItemQueue<ProcedureInfo, typename PROCEDURE::ExpectedData>
//  {
//    static const bool Itemized = true;
//    static const bool Empty = false;
//    typedef PROCEDURE Procedure;
//  };
//  template<class PROCEDURE, class ProcedureInfo, template<class ProcedureInfo, class QueueElement> class InternalPackageQueue, template<class ProcedureInfo, class QueueElement> class InternalItemQueue, int procIdSel>
//  struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, InternalItemQueue, false, procIdSel> : public InternalPackageQueue<ProcedureInfo, typename PROCEDURE::ExpectedData>
//  {
//    static const bool Itemized = false;
//    static const bool Empty = false;
//    typedef PROCEDURE Procedure;
//  };
//  template<class PROCEDURE, class ProcedureInfo, template<class ProcedureInfo, class QueueElement> class InternalPackageQueue, template<class ProcedureInfo, class QueueElement> class InternalItemQueue>
//  struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, InternalItemQueue, true, -1> : public NoQueue<ProcedureInfo, typename PROCEDURE::ExpectedData>
//  {
//    static const bool Itemized = true;
//    static const bool Empty = true;
//    typedef PROCEDURE Procedure;
//  };
//  template<class PROCEDURE, class ProcedureInfo, template<class ProcedureInfo, class QueueElement> class InternalPackageQueue, template<class ProcedureInfo, class QueueElement> class InternalItemQueue>
//  struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, InternalItemQueue, false, -1> : public NoQueue<ProcedureInfo, typename PROCEDURE::ExpectedData>
//  {
//    static const bool Itemized = false;
//    static const bool Empty = true;
//    typedef PROCEDURE Procedure;
//  };
//  
//  template<int ProcId>
//  struct ProcIdSelector
//  {
//    static const int Id = ProcId < 0 ? -1 : ProcId;
//  };
//
//
//  template<class NEWPROCEDURE, class PROCEDUREPRESENT, class TheQueue>
//  class QueueMatchSelector
//  {
//  public:
//    __inline__ __device__ static bool enqueueInitial(TheQueue& q, const typename NEWPROCEDURE::ExpectedData & data) 
//    {  return false;  }
//
//    __inline__ __device__ static bool enqueue(TheQueue& q, const typename NEWPROCEDURE::ExpectedData& data) 
//    { return false;    }
//
//    __inline__ __device__ static int dequeue(TheQueue& q, typename NEWPROCEDURE::ExpectedData* data, int& procId, int startThread = 0)
//    { return 0; }
//    
//    __inline__ __device__ static int size()
//    { return 0; }
//  };
//
//  template< class PROCEDUREMATCH, class TheQueue>
//  class QueueMatchSelector<PROCEDUREMATCH, PROCEDUREMATCH, TheQueue>
//  {
//  public:
//    __inline__ __device__ static bool enqueueInitial(TheQueue& q, const typename PROCEDUREMATCH::ExpectedData & data) 
//    {
//      return q . template enqueueInitial<PROCEDUREMATCH>(data);
//    }
//    __inline__ __device__ static bool enqueue(TheQueue& q, typename PROCEDUREMATCH::ExpectedData const& data) 
//    { 
//      return q . template enqueue<PROCEDUREMATCH>(data);
//    }
//
//    __inline__ __device__ static int dequeue(TheQueue& q, typename PROCEDUREMATCH::ExpectedData* data, int& procId, int startThread = 0)
//    {  
//      return q . template dequeue<typename PROCEDUREMATCH::ExpectedData>(data, procId, startThread);
//    }
//
//    __inline__ __device__ static int size(const TheQueue& q)
//    {  
//      return q . size();
//    }
//  };
//
//  template<class Q, bool Empty, class Visitor>
//  class QueueVisitor
//  {
//  public:
//    __inline__ __device__ 
//    static bool visit(Q& q, Visitor& visitor)
//    {
//      return visitor. template visit<Q, typename Q::Procedure, Q::Itemized > (q);
//    }
//  };
//  template<class Q, class Visitor>
//  class QueueVisitor<Q, true, Visitor>
//  {
//  public:
//    __inline__ __device__ 
//    static bool visit(Q& q, Visitor& visitor)
//    {
//      return false;
//    }
//  };
//
//
//  template<class ProcedureInfo, template<class ProcedureInfo, class QueueElement> class InternalPackageQueue, template<class ProcedureInfo, class QueueElement> class InternalItemQueue = NoQueue>
//  class MultiQueue
//  {
//    typedef QueueSelector<typename ProcedureInfo::Procedure0, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure0::ItemInput, ProcIdSelector<ProcedureInfo::Procedure0::ProcedureId>::Id> Q0_t; Q0_t Q0;
//    typedef QueueSelector<typename ProcedureInfo::Procedure1, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure1::ItemInput, ProcIdSelector<ProcedureInfo::Procedure1::ProcedureId>::Id> Q1_t; Q1_t Q1;
//    typedef QueueSelector<typename ProcedureInfo::Procedure2, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure2::ItemInput, ProcIdSelector<ProcedureInfo::Procedure2::ProcedureId>::Id> Q2_t; Q2_t Q2;
//    typedef QueueSelector<typename ProcedureInfo::Procedure3, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure3::ItemInput, ProcIdSelector<ProcedureInfo::Procedure3::ProcedureId>::Id> Q3_t; Q3_t Q3;
//    typedef QueueSelector<typename ProcedureInfo::Procedure4, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure4::ItemInput, ProcIdSelector<ProcedureInfo::Procedure4::ProcedureId>::Id> Q4_t; Q4_t Q4;
//    typedef QueueSelector<typename ProcedureInfo::Procedure5, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure5::ItemInput, ProcIdSelector<ProcedureInfo::Procedure5::ProcedureId>::Id> Q5_t; Q5_t Q5;
//    typedef QueueSelector<typename ProcedureInfo::Procedure6, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure6::ItemInput, ProcIdSelector<ProcedureInfo::Procedure6::ProcedureId>::Id> Q6_t; Q6_t Q6;
//    typedef QueueSelector<typename ProcedureInfo::Procedure7, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure7::ItemInput, ProcIdSelector<ProcedureInfo::Procedure7::ProcedureId>::Id> Q7_t; Q7_t Q7;
//    typedef QueueSelector<typename ProcedureInfo::Procedure8, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure8::ItemInput, ProcIdSelector<ProcedureInfo::Procedure8::ProcedureId>::Id> Q8_t; Q8_t Q8;
//    typedef QueueSelector<typename ProcedureInfo::Procedure9, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure9::ItemInput, ProcIdSelector<ProcedureInfo::Procedure9::ProcedureId>::Id> Q9_t; Q9_t Q9;
//    typedef QueueSelector<typename ProcedureInfo::Procedure10, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure10::ItemInput, ProcIdSelector<ProcedureInfo::Procedure10::ProcedureId>::Id> Q10_t; Q10_t Q10;
//    typedef QueueSelector<typename ProcedureInfo::Procedure11, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure11::ItemInput, ProcIdSelector<ProcedureInfo::Procedure11::ProcedureId>::Id> Q11_t; Q11_t Q11;
//    typedef QueueSelector<typename ProcedureInfo::Procedure12, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure12::ItemInput, ProcIdSelector<ProcedureInfo::Procedure12::ProcedureId>::Id> Q12_t; Q12_t Q12;
//    typedef QueueSelector<typename ProcedureInfo::Procedure13, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure13::ItemInput, ProcIdSelector<ProcedureInfo::Procedure13::ProcedureId>::Id> Q13_t; Q13_t Q13;
//    typedef QueueSelector<typename ProcedureInfo::Procedure14, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure14::ItemInput, ProcIdSelector<ProcedureInfo::Procedure14::ProcedureId>::Id> Q14_t; Q14_t Q14;
//    typedef QueueSelector<typename ProcedureInfo::Procedure15, ProcedureInfo, InternalPackageQueue, InternalItemQueue, ProcedureInfo::Procedure15::ItemInput, ProcIdSelector<ProcedureInfo::Procedure15::ProcedureId>::Id> Q15_t; Q15_t Q15;
//
//  public:
//    __inline__ __device__ void init() 
//    {
//      Q0.init();
//      Q1.init();
//      Q2.init();
//      Q3.init();
//      Q4.init();
//      Q5.init();
//      Q6.init();
//      Q7.init();
//      Q8.init();
//      Q9.init();
//      Q10.init();
//      Q11.init();
//      Q12.init();
//      Q13.init();
//      Q14.init();
//      Q15.init();
//    }
//
//     template<class PROCEDURE>
//    __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData const& data) 
//    {
//      return
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::enqueueInitial(Q0, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::enqueueInitial(Q1, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::enqueueInitial(Q2, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::enqueueInitial(Q3, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::enqueueInitial(Q4, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::enqueueInitial(Q5, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::enqueueInitial(Q6, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::enqueueInitial(Q7, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::enqueueInitial(Q8, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::enqueueInitial(Q9, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::enqueueInitial(Q10, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::enqueueInitial(Q11, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::enqueueInitial(Q12, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::enqueueInitial(Q13, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::enqueueInitial(Q14, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::enqueueInitial(Q15, data);
//    }
//
//    template<class PROCEDURE>
//    __device__ bool enqueue(const typename PROCEDURE::ExpectedData& data) 
//    {     
//      return
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::enqueue(Q0, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::enqueue(Q1, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::enqueue(Q2, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::enqueue(Q3, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::enqueue(Q4, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::enqueue(Q5, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::enqueue(Q6, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::enqueue(Q7, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::enqueue(Q8, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::enqueue(Q9, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::enqueue(Q10, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::enqueue(Q11, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::enqueue(Q12, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::enqueue(Q13, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::enqueue(Q14, data) || 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::enqueue(Q15, data);
//    }
//
//    template<class PROCEDURE>
//    __inline__ __device__ int dequeue(typename PROCEDURE::ExpectedData* data, int& procId, int startThread = 0)
//    {
//      return 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::dequeue(Q0, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::dequeue(Q1, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::dequeue(Q2, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::dequeue(Q3, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::dequeue(Q4, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::dequeue(Q5, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::dequeue(Q6, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::dequeue(Q7, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::dequeue(Q8, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::dequeue(Q9, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::dequeue(Q10, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::dequeue(Q11, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::dequeue(Q12, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::dequeue(Q13, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::dequeue(Q14, data) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::dequeue(Q15, data);
//    }
//
//    template<class PROCEDURE>
//    __inline__ __device__ int size() const
//    {
//      return 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::size(Q0) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::size(Q1) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::size(Q2) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::size(Q3) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::size(Q4) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::size(Q5) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::size(Q6) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::size(Q7) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::size(Q8) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::size(Q9) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::size(Q10) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::size(Q11) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::size(Q12) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::size(Q13) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::size(Q14) + 
//      QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::size(Q15);
//    }
//
//    static std::string name()
//    {
//
//      return 
//      Q0_t::name() + 
//      Q1_t::name() + 
//      Q2_t::name() + 
//      Q3_t::name() + 
//      Q4_t::name() + 
//      Q5_t::name() + 
//      Q6_t::name() + 
//      Q7_t::name() + 
//      Q8_t::name() + 
//      Q9_t::name() + 
//      Q10_t::name() + 
//      Q11_t::name() + 
//      Q12_t::name() + 
//      Q13_t::name() + 
//      Q14_t::name() + 
//      Q15_t::name();
//    }
//
//    template<class Visitor>
//    __inline__ __device__ bool visitAll(Visitor& visit)
//    {
//      if(QueueVisitor<Q0_t, Q0_t::Empty, Visitor>::visit(Q0, visit)) return true;
//      if(QueueVisitor<Q1_t, Q1_t::Empty, Visitor>::visit(Q1, visit)) return true;
//      if(QueueVisitor<Q2_t, Q2_t::Empty, Visitor>::visit(Q2, visit)) return true;
//      if(QueueVisitor<Q3_t, Q3_t::Empty, Visitor>::visit(Q3, visit)) return true;
//      if(QueueVisitor<Q4_t, Q4_t::Empty, Visitor>::visit(Q4, visit)) return true;
//      if(QueueVisitor<Q5_t, Q5_t::Empty, Visitor>::visit(Q5, visit)) return true;
//      if(QueueVisitor<Q6_t, Q6_t::Empty, Visitor>::visit(Q6, visit)) return true;
//      if(QueueVisitor<Q7_t, Q7_t::Empty, Visitor>::visit(Q7, visit)) return true;
//      if(QueueVisitor<Q8_t, Q8_t::Empty, Visitor>::visit(Q8, visit)) return true;
//      if(QueueVisitor<Q9_t, Q9_t::Empty, Visitor>::visit(Q9, visit)) return true;
//      if(QueueVisitor<Q10_t, Q10_t::Empty, Visitor>::visit(Q10, visit)) return true;
//      if(QueueVisitor<Q11_t, Q11_t::Empty, Visitor>::visit(Q11, visit)) return true;
//      if(QueueVisitor<Q12_t, Q12_t::Empty, Visitor>::visit(Q12, visit)) return true;
//      if(QueueVisitor<Q13_t, Q13_t::Empty, Visitor>::visit(Q13, visit)) return true;
//      if(QueueVisitor<Q14_t, Q14_t::Empty, Visitor>::visit(Q14, visit)) return true;
//      if(QueueVisitor<Q15_t, Q15_t::Empty, Visitor>::visit(Q15, visit)) return true;
//      return false;
//    }
//  };
//
//
//  template<class ProcedureInfo, class QueueElement, template<class ProcedureInfo, class QueueElement> class InternalPackageQueue, template<class ProcedureInfo, class QueueElement> class InternalItemQueue = NoQueue >
//  class PerProcedureQueue : public ::Queue<ProcedureInfo, QueueElement> 
//  {
//    typedef MultiQueue<ProcedureInfo, InternalPackageQueue, InternalItemQueue>  MyMultiQueue;
//    MyMultiQueue queues;
//
//    class Visitor
//    {
//      int _num;
//      int& _haveSomething;
//      int& _hasSomething;
//      int& _procId;
//      int _startThread;
//      QueueElement* _data;
//      const int _itemizedThreshold;
//    public:
//      __inline__ __device__ Visitor(int num, int& haveSomething, int& hasSomething, int& procId, int startThread, QueueElement* data, int minItems) : 
//          _num(num), _haveSomething(haveSomething), _hasSomething(hasSomething), _procId(procId), _startThread(startThread), _data(data), _itemizedThreshold(minItems)
//      { }
//      template<class Queue, class Procedure, bool Itemized>
//      __inline__ __device__ bool visit(Queue& q)
//      {
//        if(Itemized)
//        {
//          __shared__ int size;
//          if(threadIdx.x == _startThread)
//            size = q.size();
//          Softshell::syncthreads(2, _num);
//          if(size >= _itemizedThreshold)
//          {
//            _hasSomething = q . dequeue(reinterpret_cast<typename Procedure::ExpectedData*>(_data), _procId, _startThread);
//            if(threadIdx.x == _startThread)
//              _haveSomething = _hasSomething;
//            Softshell::syncthreads(2, _num);
//            return _haveSomething;
//          }
//          return false;
//        }
//        else
//        {
//          _hasSomething = q.dequeue(reinterpret_cast<typename Procedure::ExpectedData*>(_data), _procId, _startThread);
//          _haveSomething = _hasSomething;
//          return _hasSomething;
//        }
//      }
//    };
//
//  public:
//    static std::string name()
//    {
//      std::stringstream sstr;
//      sstr << "DistributedPerProcedure[" << MyMultiQueue::name() << "]";
//      return sstr.str();
//    }
//
//    __inline__ __device__ void init() 
//    {
//      queues.init();
//    }
//
//    template<class PROCEDURE>
//    __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData const& data) 
//    {
//      return queues. template enqueueInitial<PROCEDURE>(data);
//    }
//
//    template<class PROCEDURE>
//    __device__ bool enqueue(typename PROCEDURE::ExpectedData const& data) 
//    {        
//      return queues. template enqueue<PROCEDURE>(data);
//    }
//
//    __inline__ __device__ int dequeue(QueueElement* data, int& procId, int startThread = 0)
//    {
//      if(threadIdx.x < startThread)
//        return 0;
//
//      __shared__ int haveSomething;
//      haveSomething = 0;
//      int hasSomething;
//      int num = blockDim.x - startThread;
//
//      Softshell::syncthreads(2, num);
//      
//      
//
//      Visitor visitor(num, haveSomething, hasSomething, procId, startThread, data, blockDim.x);
//      if(queues. template visitAll<Visitor>(visitor))
//        return hasSomething;
//      Visitor visitor2(num, haveSomething, hasSomething, procId, startThread, data, 0);
//      if(queues. template visitAll<Visitor>(visitor2))
//        return hasSomething;
//      return 0;
//
//      //#pragma unroll
//      //for(int qId = 0; qId < Procedures && !haveSomething; ++qId)
//      //{
//      //  __syncthreads();
//      //  int hasData = queues[qId]. template dequeue<DATA>(data, procId);
//      //  if(threadIdx.x == 0)
//      //    haveSomething = hasData;
//      //  __syncthreads();
//      //  if(haveSomething)
//      //    return hasData;
//      //}
//      //return 0;
//    }
//  };
//
//
//  template<class ProcedureInfo, class QueueElement, int MaxBlocks, template<class ProcedureInfo, class QueueElement> class InternalQueue_T>
//  class PerBlockQueue : public ::Queue<ProcedureInfo, QueueElement>
//  {
//    typedef InternalQueue_T<ProcedureInfo, QueueElement> InternalQueue;
//    InternalQueue queues[MaxBlocks];
//  public:
//    static std::string name()
//    {
//      std::stringstream sstr;
//      sstr << "DistributedPerBlock[" << InternalQueue::name() << "]x" << MaxBlocks;
//      return sstr.str();
//    }
//
//    __inline__ __device__ void init() 
//    {
//      for(int i = 0; i < MaxBlocks; ++i)
//        queues[i].init();
//    }
//
//    template<class PROCEDURE>
//    __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData const& data) 
//    {
//      int qId = random::rand() % MaxBlocks;
//      return queues[qId]. template enqueue<PROCEDURE>(data);
//    }
//
//    template<class PROCEDURE>
//    __device__ bool enqueue(typename PROCEDURE::ExpectedData const& data) 
//    {        
//      return queues[blockIdx.x % MaxBlocks]. template enqueue<PROCEDURE>(data);
//    }
//
//    __inline__ __device__ int dequeue(QueueElement* data, int& procId, int startThread = 0)
//    {
//      return queues[ blockIdx.x % MaxBlocks]. dequeue(data, procId);
//    }
//  };
//
//
//  template<class ProcedureInfo, class QueueElement, uint MaxBlocks, template<class ProcedureInfo, class QueueElement> class InternalQueue_T>
//  class PerBlockStealing : public ::Queue<ProcedureInfo, QueueElement>
//  {
//    typedef InternalQueue_T<ProcedureInfo, QueueElement> InternalQueue;
//    InternalQueue queues[MaxBlocks];
//  public:
//    typedef typename ProcedureInfo::QueueDataContainer QueueElement;
//    static std::string name()
//    {
//      std::stringstream sstr;
//      sstr << "DistributedPerBlockStealing[" << InternalQueue::name() << "]x" << MaxBlocks;
//      return sstr.str();
//    }
//
//    __inline__ __device__ void init() 
//    {
//      for(int i = 0; i < MaxBlocks; ++i)
//        queues[i].init();
//    }
//
//    template<class PROCEDURE>
//    __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData const& data) 
//    {
//      int qId = random::rand() % MaxBlocks;
//      return queues[qId]. template enqueue<PROCEDURE>(data);
//    }
//
//    template<class PROCEDURE>
//    __device__ bool enqueue(typename PROCEDURE::ExpectedData const& data) 
//    {        
//      return queues[blockIdx.x % MaxBlocks]. template enqueue<PROCEDURE>(data);
//    }
//
//    __inline__ __device__ int dequeue(QueueElement* data, int& procId, int startThread = 0)
//    {
//      __shared__ int haveSomething;
//      haveSomething = 0;
//      int b = blockIdx.x % MaxBlocks;
//      __syncthreads();
//
//      #pragma unroll
//      for(int i = 0; i < MaxBlocks && !haveSomething; ++i)
//      {
//        __syncthreads();
//        int hasData = queues[b].dequeue(data, procId);
//        if(threadIdx.x == 0)
//          haveSomething = hasData;
//        __syncthreads();
//        if(haveSomething)
//          return hasData;
//        b = (b + 1) % MaxBlocks;
//      }
//      return 0;
//    }
//  };
//
//
//  template<class ProcedureInfo, class QueueElement, uint MaxBlocks, template<class ProcedureInfo, class QueueElement> class InternalQueue_T, bool AssertOnOverflow = true>
//  class FixedPerBlockDonating : public ::Queue<ProcedureInfo, QueueElement>
//  {
//    typedef InternalQueue_T<ProcedureInfo,  QueueElement> InternalQueue;
//    InternalQueue queues[MaxBlocks];
//
//    template<class PROCEDURE>
//    __device__ bool enqueue(int qId, typename PROCEDURE::ExpectedData const& data) 
//    {
//      #pragma unroll
//      for(int i = 0; i < MaxBlocks; ++i, qId = (qId + 1) % MaxBlocks)
//        if(queues[qId]. template enqueue<PROCEDURE>(data))
//          return true;
//
//      if(AssertOnOverflow)
//      {
//        printf("ERROR queue out of elements!\n");
//        Softshell::trap();
//      }
//      return false;
//    }
//  public:
//    static std::string name()
//    {
//      std::stringstream sstr;
//      sstr << "DistributedPerBlockDonatingFixed[" << InternalQueue::name() << "]x" << MaxBlocks;
//      return sstr.str();
//    }
//
//    __inline__ __device__ void init() 
//    {
//      for(int i = 0; i < MaxBlocks; ++i)
//        queues[i]. init();
//    }
//
//    template<class PROCEDURE>
//    __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData const& data) 
//    {
//      int qId = random::rand() % MaxBlocks;
//      return enqueue<PROCEDURE>(qId, data);
//    }
//
//
//    template<class PROCEDURE>
//    __device__ bool enqueue(typename PROCEDURE::ExpectedData const& data) 
//    {
//      int qId = blockIdx.x % MaxBlocks;
//      return enqueue<PROCEDURE>(qId, data);
//    }
//
//    __inline__ __device__ int dequeue(QueueElement* data, int& procId, int startThread = 0)
//    {
//      return queues[ blockIdx.x % MaxBlocks]. dequeue(data, procId);
//    }
//  };
//
//
//  template<class ProcedureInfo, class QueueElement, uint MaxBlocks, template<class ProcedureInfo, class QueueElement> class InternalQueue_T, int DonateThreshold = 10, int DonateProbability = 5, bool AssertOnOverflow = true>
//  class RandomPerBlockDonating : public ::Queue<ProcedureInfo, QueueElement>
//  {
//    typedef InternalQueue_T<ProcedureInfo, QueueElement> InternalQueue;
//    InternalQueue queues[MaxBlocks];
//
//    template<class PROCEDURE, class DATA>
//    __device__ bool enqueue(int qId, const DATA& data) 
//    {
//      #pragma unroll
//      for(int i = 0; i < MaxBlocks; ++i, qId = (qId + 1) % MaxBlocks)
//        if(queues[qId]. template enqueue<PROCEDURE>(data))
//          return true;
//
//      if(AssertOnOverflow)
//      {
//        printf("ERROR queue out of elements!\n");
//        Softshell::trap();
//      }
//      return false;
//    }
//  public:
//    static std::string name()
//    {
//      std::stringstream sstr;
//      sstr << "DistributedPerBlockDonatingRandom[" << InternalQueue::name() << "]x" << MaxBlocks << "(" << DonateThreshold << "," << DonateProbability << ")";
//      return sstr.str();
//    }
//
//    __inline__ __device__ void init() 
//    {
//      for(int i = 0; i < MaxBlocks; ++i)
//        queues[i]. init();
//    }
//
//    template<class PROCEDURE>
//    __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData const& data) 
//    {
//      int qId = random::rand() % MaxBlocks;
//      return enqueue<PROCEDURE, typename PROCEDURE::ExpectedData>(qId, data);
//    }
//
//
//    template<class PROCEDURE>
//    __device__ bool enqueue(typename PROCEDURE::ExpectedData const& data) 
//    {
//      int qId = blockIdx.x % MaxBlocks;
//      if(queues[qId].size() > DonateThreshold && random::warp_check(DonateProbability))
//        qId = random::warp_rand() % MaxBlocks;
//      return enqueue<PROCEDURE,  typename PROCEDURE::ExpectedData>(qId, data);
//    }
//
//    __inline__ __device__ int dequeue(QueueElement* data, int& procId, int startThread = 0)
//    {
//      return queues[ blockIdx.x % MaxBlocks]. dequeue(data, procId);
//    }
//  };
//}