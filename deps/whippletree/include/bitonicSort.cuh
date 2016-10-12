
/*
* file created by    Markus Steinberger / steinberger ( at ) icg.tugraz.at
*
* modifications by
*/

#ifndef SOFTSHELL_TOOLS_BITONICSOURT_INCLUDED
#define SOFTSHELL_TOOLS_BITONICSOURT_INCLUDED

#include "common.cuh"

namespace Softshell
{
  namespace Sort
  {
    template<class Key, class Value>
    __device__ inline void bitonic_comp(volatile Key& key_a, volatile Key& key_b,
                                        volatile Value& val_a, volatile Value& val_b,
                                        bool dir)
    {
      if((key_a != key_b) && (key_a > key_b) == dir )
      {
        //swap
        Key kT = key_a;
        key_a = key_b;
        key_b = kT;

        Value vT = val_a;
        val_a = val_b;
        val_b = vT;
      }
    }


    template<class Key, class Value, bool Dir>
    __device__ void bitonic(volatile Key* keys, volatile Value* values, uint linId, uint elements)
    {
      if(linId < elements / 2)
      {
        for(uint size = 2; size < elements; size <<= 1)
        {
          //bitonic merge
          bool d = Dir ^ ( (linId & (size / 2)) != 0 );
          for(uint stride = size / 2; stride > 0; stride >>= 1)
          {
            syncthreads(1, elements/2);
            uint pos = 2 * linId - (linId & (stride - 1));
            bitonic_comp(keys[pos], keys[pos + stride],
                         values[pos], values[pos + stride],
                         d);
          }
        }

        //final merge
        for(uint stride = elements / 2; stride > 0; stride >>= 1)
        {
            syncthreads(1, elements/2);
            uint pos = 2 * linId - (linId & (stride - 1));
            bitonic_comp(keys[pos], keys[pos + stride],
                         values[pos], values[pos + stride],
                         Dir);
        }
      }
      syncthreads(1, elements/2);
    }
  }
}


#endif  // SOFTSHELL_TOOLS_BITONICSOURT_INCLUDED
