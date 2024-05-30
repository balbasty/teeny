#include "../include/miniten/tuple.h"

using namespace miniten;
using namespace miniten::meta;

int main()
{
    using Foo1 = Vector<int, 1, 2, 3>;
    show<Foo1>();
    newline();
    show("- length: ");
    show(Foo1::Length);
    newline();
    show("- first:  ");
    show(Foo1::First);
    newline();
    show("- last:   ");
    show(Foo1::Last);
    newline();

    using Foo2 = Vector<int, 4, 5, 6>;
    show<Foo2>();
    newline();

    using Foo12 = typename Cat<Foo1, Foo2>::Type;
    show<Foo12>();
    newline();

    using Bar = Tuple<int, long, float>;
    show<Bar>();
    newline();

    using BarDel = typename DelItem<2, Bar>::Type;
    show<BarDel>();
    newline();

}
