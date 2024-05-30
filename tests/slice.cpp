#include "../include/miniten/meta/vector.h"
#include "../include/miniten/meta/index.h"
#include "../include/miniten/meta/show.h"


using namespace miniten;
using namespace miniten::meta;

int main()
{
    using Slice1 = Slice<0, 1>;
    show<Slice1>();
    newline();

    show("AsVector: ");
    show<Slice1::AsVector<10> >();
    newline();

    using Slice2 = Slice<0, 2>;
    show<Slice2>();
    newline();

    show("AsVector: ");
    show<Slice2::AsVector<10> >();
    newline();

    using SliceM1 = Slice<-1, 10>;
    show<SliceM1>();
    newline();

    show("AsVector: ");
    show<SliceM1::AsVector<10> >();
    newline();

    using SliceM2 = Slice<-2, 10>;
    show<SliceM2>();
    newline();

    show("AsVector: ");
    show<SliceM2::AsVector<10> >();
    newline();

    show("---");
    newline();

    show<Slice<0, 10, 2>::AsVector<10> >();
    newline();
    show<Slice<1, 10, 2>::AsVector<10> >();
    newline();

    show<Slice<0, 9, 2>::AsVector<10> >();
    newline();

    show<Slice<1, 9, 2>::AsVector<10> >();
    newline();

    show<Slice<0, 0>::AsVector<10> >();
    newline();
    show<Slice<10, 10>::AsVector<10> >();
    newline();

    show("---");
    newline();

    using SmartSlice1 = SmartSlice<Long<0>, Long<1> >;
    show<SmartSlice1>();
    newline();

    using SmartSlice2 = SmartSlice<Long<0>, Long<2> >;
    show<SmartSlice1>();
    newline();
}
