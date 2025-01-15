#ifndef MINITEN_META_PACK_SHOW_H
#define MINITEN_META_PACK_SHOW_H
#include "../_core/defines.h"
#include "../show.h"
#include "pack.h"

namespace miniten {

template <class X0, class... X>
struct Show< meta::Pack<X0, X...> > {
    using Type = meta::Pack<X0, X...>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Pack<");
        showValues();
        miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {
        miniten::show<X0>(); miniten::show(", ");
        Show<meta::Pack<X...>>::showValues();
    }

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};


template <class X>
struct Show< meta::Pack<X> > {
    using Type = meta::Pack<X>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Pack<");
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
        show();
    }
};

template <>
struct Show< meta::Pack<> > {
    using Type = meta::Pack<>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Pack<>");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {}


    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

} // namespace miniten

#endif // MINITEN_META_PACK_SHOW_H
