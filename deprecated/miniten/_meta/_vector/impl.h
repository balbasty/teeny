/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements a compile-time "vector of values"                  **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META__VECTOR_IMPL
#define MINITEN__META__VECTOR_IMPL
#include <miniten/_core/defines.h>
#include <miniten/_core/types.h>
#include <miniten/_meta/_vector/decl.h>     // Vector, NVector
#include <miniten/_meta/_packapi/decl.h>    // Cat
#include <miniten/_meta/_math/decl.h>       // CountValues

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

/* ------------------------------------------------------------------ *
 *     Repeat                                                         *
 * ------------------------------------------------------------------ */

template <long N, typename T, T Val>
struct _NVector {
    using Type = Cat<NVector<N-1, T, Val>, Vector<T, Val>>;
};

template <typename T, T Val>
struct _NVector<0, T, Val> {
    using Type = Vector<T>;
};

/* ------------------------------------------------------------------ *
 *     Vector Specialization                                          *
 * ------------------------------------------------------------------ */

struct AnyVectorBase {};

template <typename T>
struct VectorBase: public AnyVectorBase {
    using item_t = T;
};

/* --- A compile-time tuple of values with the same type ------------ */
template <typename T, T... X>
struct Vector: public VectorBase<T> {};

/* --- 1+ elements -------------------------------------------------- */
template <typename T, T X0, T... X>
struct Vector<T, X0, X...>: public VectorBase<T> {

    /** * * * * * * * * * * * * **
     ** Static types and values **
     ** * * * * * * * * * * * * **/

    using this_t = Vector<T, X0, X...>;
    using next_t = Vector<T, X...>;
    using item_t = T;
    MINIDEF(S,CX) size_t Length = CountValues<T, X0, X...>::Value;

    /** * * * * * * * * * **
     ** Constexpr methods **
     ** * * * * * * * * * **/

    MINIDEF(H,D,CX) size_t     length()    const { return Length;}
    MINIDEF(H,D,CX) bool       empty()     const { return Length == 0; }

    template <class I>              MINIDEF(H,D,CX) T                            getValue(I)         const { return GetValue<this_t,I>::Value; }
                                    MINIDEF(H,D,CX) T                            getFirstValue()     const { return GetFirstValue<this_t>::Value; }
                                    MINIDEF(H,D,CX) T                            getLastValue()      const { return GetLastValue<this_t>::Value; }

    template <class I>              MINIDEF(H,D,CX) Get<this_t,I>                get(I)              const { return Get<this_t,I>(); }
    template <class N>              MINIDEF(H,D,CX) GetFirst<this_t,N::Value>    getFirst(N)         const { return GetFirst<this_t,N::Value>(); }
                                    MINIDEF(H,D,CX) GetFirst<this_t>             getFirst()          const { return GetFirst<this_t>(); }
    template <class N>              MINIDEF(H,D,CX) GetLast<this_t,N::Value>     getLast(N)          const { return GetLast<this_t,N::Value>(); }
                                    MINIDEF(H,D,CX) GetLast<this_t>              getLast()           const { return GetLast<this_t>(); }

    template <class I>              MINIDEF(H,D,CX) Del<this_t,I>                del(I)              const { return Del<this_t,I>(); }
    template <class N>              MINIDEF(H,D,CX) DelFirst<this_t,N::Value>    delFirst(N)         const { return DelFirst<this_t,N::Value>(); }
    template <class N>              MINIDEF(H,D,CX) DelLast<this_t,N::Value>     delLast(N)          const { return DelLast<this_t,N::Value>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) SetFrom<this_t,I,M...>       setFrom(I, M...)    const { return SetFrom<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) SetFirstFrom<this_t,M...>    setFirstFrom(M...)  const { return SetFirstFrom<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) SetLastFrom<this_t,M...>     setLastFrom(M...)   const { return SetLastFrom<this_t,M...>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) Set<this_t,I,M...>           set(I, M...)        const { return Set<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) SetFirst<this_t,M...>        setFirst(M...)      const { return SetFirst<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) SetLast<this_t,M...>         setLast(M...)       const { return SetLast<this_t,M...>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) InsertFrom<this_t,I,M...>    insertFrom(I, M...) const { return InsertFrom<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) PrependFrom<this_t,M...>     prependFrom(M...)   const { return PrependFrom<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) AppendFrom<this_t,M...>      appendFrom(M...)    const { return AppendFrom<this_t,M...>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) Insert<this_t,I,M...>        insert(I, M...)     const { return Insert<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Prepend<this_t,M...>         prepend(M...)       const { return Prepend<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Append<this_t,M...>          append(M...)        const { return Append<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Extend<this_t,M...>          extend(M...)        const { return Extend<this_t,M...>(); }
};

/* --- 1 element ---------------------------------------------------- */
template <typename T, T X>
struct Vector<T, X>: public VectorBase<T> {

    /** * * * * * * * * * * * * **
     ** Static types and values **
    ** * * * * * * * * * * * * **/

    using this_t = Vector<T, X>;
    using next_t = Vector<T>;
    using item_t = T;
    MINIDEF(S,CX) size_t Length = 1;
    MINIDEF(S,CX) T      Value  = X;

    /** * * * * * * * * * **
     ** Constexpr methods **
     ** * * * * * * * * * **/

    MINIDEF(H,D,CX) size_t     length()    const { return Length; }
    MINIDEF(H,D,CX) operator   T()         const { return Value; }

    template <class I>              MINIDEF(H,D,CX) T                           getValue(I)         const { return GetValue<this_t,I>::Value; }
                                    MINIDEF(H,D,CX) T                           getFirstValue()     const { return GetFirstValue<this_t>::Value; }
                                    MINIDEF(H,D,CX) T                           getLastValue()      const { return GetLastValue<this_t>::Value; }
    template <class I>              MINIDEF(H,D,CX) Get<this_t,I>               get(I)              const { return Get<this_t,I>(); }
    template <class N>              MINIDEF(H,D,CX) GetFirst<this_t,N::Value>   getFirst(N)         const { return GetFirst<this_t,N::Value>(); }
                                    MINIDEF(H,D,CX) this_t                      getFirst()          const { return this_t(); }
    template <class N>              MINIDEF(H,D,CX) GetLast<this_t,N::Value>    getLast(N)          const { return GetLast<this_t,N::Value>(); }
                                    MINIDEF(H,D,CX) this_t                      getLast()           const { return this_t(); }

    template <class I>              MINIDEF(H,D,CX) Del<this_t,I>               del(I)              const { return Del<this_t,I>(); }
    template <class N>              MINIDEF(H,D,CX) DelFirst<this_t,N::Value>   delFirst(N)         const { return DelFirst<this_t,N::Value>(); }
    template <class N>              MINIDEF(H,D,CX) DelLast<this_t,N::Value>    delLast(N)          const { return DelLast<this_t,N::Value>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) SetFrom<this_t,I,M...>      setFrom(I, M...)    const { return SetFrom<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) SetFirstFrom<this_t,M...>   setFirstFrom(M...)  const { return SetFirstFrom<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) SetLastFrom<this_t,M...>    setLastFrom(M...)   const { return SetLastFrom<this_t,M...>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) Set<this_t,I,M...>          set(I, M...)        const { return Set<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) SetFirst<this_t,M...>       setFirst(M...)      const { return SetFirst<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) SetLast<this_t,M...>        setLast(M...)       const { return SetLast<this_t,M...>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) InsertFrom<this_t,I,M...>   insertFrom(I, M...) const { return InsertFrom<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) PrependFrom<this_t,M...>    prependFrom(M...)   const { return PrependFrom<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) AppendFrom<this_t,M...>     appendFrom(M...)    const { return AppendFrom<this_t,M...>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) Insert<this_t,I,M...>       insert(I, M...)     const { return Insert<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Prepend<this_t,M...>        prepend(M...)       const { return Prepend<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Append<this_t,M...>         append(M...)        const { return Append<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Extend<this_t,M...>         extend(M...)        const { return Extend<this_t,M...>(); }

};

/* --- empty -------------------------------------------------------- */
template <typename T>
struct Vector<T>: public VectorBase<T> {

    /** * * * * * * * * * * * * **
     ** Static types and values **
     ** * * * * * * * * * * * * **/

    using this_t = Vector<T>;
    using item_t = T;
    MINIDEF(S,CX) size_t Length = 0;

    /** * * * * * * * * * **
     ** Constexpr methods **
     ** * * * * * * * * * **/

    MINIDEF(H,D,CX) size_t     length()    const { return Length; }

    template <class I, class... M>  MINIDEF(H,D,CX) InsertFrom<this_t,I,M...>    insertFrom(I, M...) const { return InsertFrom<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) PrependFrom<this_t,M...>     prependFrom(M...)   const { return PrependFrom<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) AppendFrom<this_t,M...>      appendFrom(M...)    const { return AppendFrom<this_t,M...>(); }

    template <class I, class... M>  MINIDEF(H,D,CX) Insert<this_t,I,M...>        insert(I, M...)     const { return Insert<this_t,I,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Prepend<this_t,M...>         prepend(M...)       const { return Prepend<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Append<this_t,M...>          append(M...)        const { return Append<this_t,M...>(); }
    template <class... M>           MINIDEF(H,D,CX) Extend<this_t,M...>          extend(M...)        const { return Extend<this_t,M...>(); }
};

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif /// MINITEN__META__VECTOR_IMPL
