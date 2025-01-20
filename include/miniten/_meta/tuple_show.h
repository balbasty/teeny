#ifndef MINITEN_META_TUPLE_SHOW_H
#define MINITEN_META_TUPLE_SHOW_H
#include "../_core/defines.h"
#include "../show.h"
#include "tuple.h"

namespace miniten {

template <class X0, class... X>
struct Show< meta::Tuple<X0, X...> > {
    using Type = meta::Tuple<X0, X...>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Tuple<");
        showValues();
        miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {
        miniten::show<X0>(); miniten::show(", ");
        Show<meta::Tuple<X...>>::showValues();
    }

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show(); miniten::show("("); showValues(value); miniten::show(")");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues(const Type & value) {
        using NextLength = meta::SizeT<Type::Length-1>;
        miniten::show(value.getFirstValue()); miniten::show(", ");
        Show<meta::DelFirst<Type>>::showValues(value.getLast(NextLength()));
    }
};


template <class X>
struct Show< meta::Tuple<X> > {
    using Type = meta::Tuple<X>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Tuple<");
        miniten::show<X>();
        miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {
        miniten::show<X>();
    }

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show(); miniten::show("("); showValues(value); miniten::show(")");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues(const Type & value) {
        miniten::show(value.getFirstValue());
    }
};

template <>
struct Show< meta::Tuple<> > {
    using Type = meta::Tuple<>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Tuple<>");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {}

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show(); miniten::show("("); showValues(value); miniten::show(")");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues(const Type & value) {}
};

} // namespace miniten

#endif // MINITEN_META_TUPLE_SHOW_H
