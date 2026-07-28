// SCRATCH MSVC repro for #296 -- NOT a real teeny feature test. Isolates the
// overload-set shape of tensor::sum<Acc,Axes...>/sum(Tags...) without any CCCL
// dependency, to nail down exactly which MSVC diagnostic fires and on which
// candidate. Delete before merging; see #296 for context.
#include <type_traits>

template <bool V> using EnableIfT = std::enable_if_t<V, int>;

struct S {
    // (1) full reduction: no axes, optional explicit Acc
    template <class Acc = void>
    int sum() const { return 1; }

    // (2) explicit axis list only (no Acc), any number of trailing tag args
    template <long... Ax, class... Tags, EnableIfT<(sizeof...(Ax) > 0)> = 0>
    int sum(Tags... /*tags*/) const { return 2; }

    // (3) explicit Acc AND axis list
    template <class Acc, long... Ax, class... Tags, EnableIfT<(sizeof...(Ax) > 0)> = 0>
    int sum(Tags... /*tags*/) const { return 3; }

    // (4) generic trailing-tag form: deduced Acc, first real arg is Tag0 (no default axis)
    template <class Acc = void, class Tag0, class... Tags>
    int sum(Tag0 /*tag0*/, Tags... /*tags*/) const { return 4; }
};

// Narrower isolation: JUST the (1)-vs-(2) pair, nothing else.
struct S2 {
    template <class Acc = void>
    int sum() const { return 1; }

    template <long... Ax, EnableIfT<(sizeof...(Ax) > 0)> = 0>
    int sum() const { return 2; }
};

// Round 2: same 4-overload shape as S, but the ENCLOSING class is itself a
// class TEMPLATE (like tensor<T,E,L,O>), and the axis-list overloads' SFINAE
// condition depends on that enclosing class template parameter (like the real
// _TNY_RED_AXIS_IF, which gates on _md::reduced_extents<E,Ax...>::rank_dynamic()).
template <long E>  // stand-in for the tensor's own Shape/extents template param
struct S3 {
    template <bool V> using EIf = std::enable_if_t<V, int>;

    template <class Acc = void>
    int sum() const { return 1; }

    template <long... Ax, class... Tags, EIf<(sizeof...(Ax) > 0) && (E > 0)> = 0>
    int sum(Tags... /*tags*/) const { return 2; }

    template <class Acc, long... Ax, class... Tags, EIf<(sizeof...(Ax) > 0) && (E > 0)> = 0>
    int sum(Tags... /*tags*/) const { return 3; }

    template <class Acc = void, class Tag0, class... Tags>
    int sum(Tag0 /*tag0*/, Tags... /*tags*/) const { return 4; }
};

int main() {
    S s;
    S2 s2;
    S3<6> s3;
    int a = s.template sum<0>();     // want: overload (2) -> 2
    int b = s2.template sum<0>();    // want: overload (2) -> 2
    int c = s3.template sum<0>();    // want: overload (2) -> 2
    return (a == 2 && b == 2 && c == 2) ? 0 : 1;
}
