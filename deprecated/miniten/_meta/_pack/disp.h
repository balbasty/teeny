#ifndef MINITEN__META__PACK_DISP
#define MINITEN__META__PACK_DISP
#include <miniten/_core/defines.h>
#include <miniten/disp.h>
#include <miniten/_meta/_pack/decl.h>

NAMESPACE_BEGIN(miniten)

template <class X0, class... X>
struct Display< meta::Pack<X0, X...> > {
    using Type = meta::Pack<X0, X...>;

    MINIDEF(H,D,S,I,CX)
    const char * srepr() {
        return "Pack<...>";
    }

    MINIDEF(H,D,S,I,CX) const char * srepr(const Type & value)
    {
        return srepr();
    }

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Pack<");
        dispValues();
        miniten::disp(">");
    }

    MINIDEF(H,D,S,I)
    void dispValues() {
        miniten::disp<X0>(); miniten::disp(", ");
        Display<meta::Pack<X...>>::dispValues();
    }

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};


template <class X>
struct Display< meta::Pack<X> > {
    using Type = meta::Pack<X>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Pack<");
        miniten::disp<X>();
        miniten::disp(">");
    }

    MINIDEF(H,D,S,I)
    void dispValues() {
        miniten::disp<X>();
    }

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

template <>
struct Display< meta::Pack<> > {
    using Type = meta::Pack<>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Pack<>");
    }

    MINIDEF(H,D,S,I)
    void dispValues() {}

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

NAMESPACE_END(miniten)

#endif // MINITEN__META__PACK_DISP
