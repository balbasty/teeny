#ifndef MINITEN_META_SHOW_H
#define MINITEN_META_SHOW_H
#include "../defines.h"
#include "../show.h"

namespace miniten {

namespace meta {
    template <class... T>       struct Tuple;
    template <class T, T... N>  struct Vector;

    template <class Start, class Stop, class Step>  struct SmartSlice;
    template <long  Start, long  Stop, long  Step>  struct Slice;
}

// template <typename... T>
// struct Show< meta::Tuple<T...> > {
//     using Type = meta::Tuple<T...>;

//     MINITEN_HOSTDEVICE static inline
//     void show()
//     {
//         Type::Show();
//     }

//     MINITEN_HOSTDEVICE static inline
//     void show(const Type & value)
//     {
//         Type::Show(value);
//     }
// };


// template <typename T, T... N>
// struct Show< meta::Vector<T, N...> > {
//     using Type = meta::Vector<T, N...>;

//     MINITEN_HOSTDEVICE static inline
//     void show() {
//         Type::Show();
//     }

//     MINITEN_HOSTDEVICE static inline
//     void show(const Type & value)
//     {
//         Type::Show(value);
//     }
// };

template <long Start, long Stop, long Step>
struct Show< meta::Slice<Start, Stop, Step> > {
    using Type = meta::Slice<Start, Stop, Step>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Slice<");
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
struct Show< meta::SmartSlice<Start, Stop, Step> > {
    using Type = meta::SmartSlice<Start, Stop, Step>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("SmartSlice<");
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

template <>
struct Show< meta::None > {
    using Type = meta::None;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("None");
    }

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

} // namespace miniten

#endif // MINITEN_META_SHOW_H
