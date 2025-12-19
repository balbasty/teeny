#ifndef MINITEN__TUPLE_DISP
#define MINITEN__TUPLE_DISP
#include <miniten/_core/defines.h>
#include <miniten/disp.h>
#include <miniten/_tuple/decl.h>

NAMESPACE_BEGIN(miniten)

template <class X0, class... X>
struct Display< Tuple<X0, X...> > {
    using Type = Tuple<X0, X...>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Tuple<");
        dispValues();
        miniten::disp(">");
    }

    MINIDEF(H,D,S,I)
    void dispValues() {
        miniten::disp<X0>(); miniten::disp(", ");
        Display<Tuple<X...>>::dispValues();
    }

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp(); miniten::disp("("); dispValues(value); miniten::disp(")");
    }

    MINIDEF(H,D,S,I)
    void dispValues(const Type & value) {
        using NextLength = meta::SizeT<Type::Length-1>;
        miniten::disp(value.getFirstValue()); miniten::disp(", ");
        Display<meta::DelFirst<Type>>::dispValues(value.getLast(NextLength()));
    }
};


template <class X>
struct Display< Tuple<X> > {
    using Type = Tuple<X>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Tuple<");
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
        disp(); miniten::disp("("); dispValues(value); miniten::disp(")");
    }

    MINIDEF(H,D,S,I)
    void dispValues(const Type & value) {
        miniten::disp(value.getFirstValue());
    }
};

template <>
struct Display< Tuple<> > {
    using Type = Tuple<>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Tuple<>");
    }

    MINIDEF(H,D,S,I)
    void dispValues() {}

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
        miniten::disp("(");
        dispValues(value);
        miniten::disp(")");
    }

    MINIDEF(H,D,S,I)
    void dispValues(const Type & value) {}
};

NAMESPACE_END(miniten)

#endif // MINITEN__TUPLE_DISP
