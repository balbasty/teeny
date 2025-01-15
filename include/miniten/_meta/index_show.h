#ifndef MINITEN_META_INDEX_SHOW_H
#define MINITEN_META_INDEX_SHOW_H
#include "../_core/defines.h"
#include "../show.h"
#include "index.h"

namespace miniten {

template <ptrdiff_t Start, ptrdiff_t Stop, ptrdiff_t Step>
struct Show< meta::SimpleSlice<Start, Stop, Step> > {
    using Type = meta::SimpleSlice<Start, Stop, Step>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("SimpleSlice<");
        miniten::show(Start); miniten::show(", ");
        miniten::show(Stop);  miniten::show(", ");
        miniten::show(Step);  miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

template <class Start, class Stop, class Step>
struct Show< meta::Slice<Start, Stop, Step> > {
    using Type = meta::Slice<Start, Stop, Step>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Slice<");
        miniten::show<Start>(); miniten::show(", ");
        miniten::show<Stop>();  miniten::show(", ");
        miniten::show<Step>();  miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

} // namespace miniten

#endif // MINITEN_META_INDEX_SHOW_H
