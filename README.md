OperatorGraph
=============


**OperatorGraph** is the reference implementation for the ideas exposed in the paper [Representing and Scheduling Procedural Generation using Operator Graphs](http://pedroboechat.com/publications/operator_graph.pdf).


[![ScreenShot](http://pedroboechat.com/images/OperatorGraph-video-thumbnail.png)](https://www.youtube.com/embed/CvAlSffwB18?list=PLgV_NS3scu1yDnjMd8m-hLoRgG8Ql7xWN)

It's essentially a toolkit that offers an end-to-end solution to compile shape grammars as programs that efficiently run on CUDA enabled GPUs.

This toolkit consists of: 
- a shape grammar interpreter,
- a C++/CUDA library and
- an execution auto-tuner.


The implemented shape grammar - __PGA-shape__ - is a rule-based language that enables users to express sequences of modeling operations in a high level of abstraction. 


__PGA-shape__ can be used as a C++/CUDA idiom or as a domain specific language (DSL). For example, to model a [Menger sponge](https://en.wikipedia.org/wiki/Menger_sponge),
you could write the following grammar in __PGA-shape__  C++/CUDA:
            
        struct Rules : T::List <
            /* SubXYZ */ Proc < Box, Subdivide<DynParam<0>, T::Pair< DynParam<1>, DCall<0>>, T::Pair< DynParam<2>, DCall<1>>, T::Pair< DynParam<3>, DCall<2>>>, 1>,
            /* Discard */ Proc < Box, Discard, 1>,
            /* A */ Proc < Box, IfSizeLess< DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>, 1>,
            /* B */ Proc < Box, Generate< false, 1 /*instanced triangle mesh*/, DynParam<0>>, 1>,
        > {};
            
or the equivalent grammar in the __PGA-shape__  DSL:
            
        axiom Box A;

        terminal B (1,0);

        A = IfSizeLess(X, 0.111) { B | SubX };
        ZDiscard = SubDiv(Z) { -1: A | -1: Discard() | -1: A };
        YDiscard = SubDiv(Y) { -1: ZDiscard | -1: Discard() | -1: ZDiscard };
        SubZ = SubDiv(Z) { -1: A | -1: A | -1: A };
        SubY = SubDiv(Y) { -1: SubZ | -1: ZDiscard | -1: SubZ };
        SubX = SubDiv(X) { -1: SubY | -1: YDiscard | -1: SubY }

Both resulting in the following Menger sponge:

![Menger Sponge](http://pedroboechat.com/images/operator-graph-menger-sponge.png)

Grammars written in the C++/CUDA idiom can be linked to OpenGL/Direct3D applications,
while grammars written in the DSL can be embedded with the interpreter. 
The difference between the two methods is that with C++/CUDA the structure of the grammars directly influences the GPU scheduling, while with the DSL the execution on the GPU is always scheduled the same way, independently of the grammar structure.


Grammars written with __PGA-shape__  DSL can be analyzed by the auto-tunner and be optimized for GPU execution.
The auto-tuner translates the DSL code to an intermediary representation - the __operator graph__ - and then exploits the graph structure 
to find the best GPU scheduling for this grammar. 
When the best scheduling is found, the auto-tuner translates back the __operator graph__ into C++/CUDA code.
