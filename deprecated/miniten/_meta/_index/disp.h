#ifndef MINITEN__META__INDEX_DISP
#define MINITEN__META__INDEX_DISP
#include <miniten/_core/defines.h>
#include <miniten/disp.h>
#include <miniten/_meta/_index/decl.h>

NAMESPACE_BEGIN(miniten)

template <
    ptrdiff_t Start,
    ptrdiff_t Stop,
    ptrdiff_t Step
>
struct Display< meta::SimpleSlice<Start, Stop, Step> > {
    using Type = meta::SimpleSlice<Start, Stop, Step>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("SimpleSlice<");
        miniten::disp(Start); miniten::disp(", ");
        miniten::disp(Stop);  miniten::disp(", ");
        miniten::disp(Step);  miniten::disp(">");
    }

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

template <
    class Start,
    class Stop,
    class Step
>
struct Display< meta::Slice<Start, Stop, Step> > {
    using Type = meta::Slice<Start, Stop, Step>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Slice<");
        miniten::disp<Start>(); miniten::disp(", ");
        miniten::disp<Stop>();  miniten::disp(", ");
        miniten::disp<Step>();  miniten::disp(">");
    }

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

NAMESPACE_END(miniten)

#endif // MINITEN__META__INDEX_DISP
