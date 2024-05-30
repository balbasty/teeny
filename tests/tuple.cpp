#include "../include/miniten/tuple.h"
#include "../include/miniten/show.h"


using namespace miniten;
using miniten::meta::Long;
using miniten::meta::SameType;


int main()
{

    using Tup0 = Tuple<>;
    show<Tup0>();
    newline();

    using Tup1 = Tuple< Long<0> >;
    show<Tup1>();
    newline();

    using Tup2 = Tuple< Long<0>, Long<1> >;
    show<Tup2>();
    newline();

    using Tup3 = Tuple< Long<0>, Long<1>, Long<2> >;
    show<Tup3>();
    newline();

    show("DelFirstItem: ");
    show<Tup3::DelFirstItem>();
    newline();
    static_assert(
        SameType<
            Tup3::DelFirstItem,
            Tuple< Long<1>, Long<2> >
        >::Value,
        "Tuple::DelFirstItem"
    );

    show("DelLastItem:  ");
    show<Tup3::DelLastItem>();
    newline();
    static_assert(
        SameType<
            Tup3::DelLastItem,
            Tuple< Long<0>, Long<1> >
        >::Value,
        "Tuple::DelLastItem"
    );

    show("Del<1>:       ");
    show<Tup3::Del<1> >();
    newline();
    static_assert(
        SameType<
            Tup3::Del<1> ,
            Tuple< Long<0>, Long<2> >
         >::Value,
        "Tuple::Del<1> "
    );

    show("Del<-2>:      ");
    show<Tup3::Del<-2> >();
    newline();
    static_assert(
        SameType<
            Tup3::Del<-2> ,
            Tuple< Long<0>, Long<2> >
        >::Value,
        "Tuple::Del<-2> "
    );

    show("GetFirstItem: ");
    show<Tup3::GetFirstItem>();
    newline();
    static_assert(
        SameType<
            Tup3::GetFirstItem,
            Long<0>
        >::Value,
        "Tuple::GetFirstItem "
    );

    show("GetLastItem:  ");
    show<Tup3::GetLastItem>();
    newline();
    static_assert(
        SameType<
            Tup3::GetLastItem,
            Long<2>
        >::Value,
        "Tuple::GetLastItem "
    );

    show("GetItem<1>:       ");
    show<Tup3::GetItem<1> >();
    newline();
    static_assert(
        SameType<
            Tup3::GetItem<1>,
            Long<1>
        >::Value,
        "Tuple::GetItem<1> "
    );

    show("GetItem<-2>:      ");
    show<Tup3::GetItem<-2> >();
    newline();
    static_assert(
        SameType<
            Tup3::GetItem<-2>,
            Long<1>
        >::Value,
        "Tuple::GetItem<-2> "
    );

    show("Get<1, 2>:    ");
    show<Tup3::Get<1, 2> >();
    newline();
    static_assert(
        SameType<
            Tup3::Get<1, 2>,
            Tuple< Long<1>, Long<2> >
        >::Value,
        "Tuple::Get<1, 2> "
    );

    show("Slice<0, 2>:  ");
    show< Tup3::Slice<0, 2> >();
    newline();
    static_assert(
        SameType<
            Tup3::Slice<0, 2>,
            Tuple< Long<0>, Long<1> >
        >::Value,
        "Tuple::Slice<0, 2>"
    );

    show("Slice<0, 0>:  ");
    show< Tup3::Slice<0, 0> >();
    newline();
    static_assert(
        SameType<
            Tup3::Slice<0, 0>,
            Tuple<>
        >::Value,
        "Tuple::Slice<0, 0>"
    );

    show("Slice<2, 2>:  ");
    show< Tup3::Slice<2, 2> >();
    newline();
    static_assert(
        SameType<
            Tup3::Slice<2, 2>,
            Tuple<>
        >::Value,
        "Tuple::Slice<2, 2>"
    );

    show("SetFirstItem< Long<9> >: ");
    show<Tup3::SetFirstItem<Long<9> > >();
    newline();
    static_assert(
        SameType<
            Tup3::SetFirstItem< Long<9> >,
            Tuple< Long<9>, Long<1>, Long<2> >
        >::Value,
         "Tuple::SetFirstItem< Long<9> > "
    );

    show("SetLastItem< Long<9> >:  ");
    show<Tup3::SetLastItem<Long<9> > >();
    newline();
    static_assert(
        SameType<
            Tup3::SetLastItem<Long<9> >,
            Tuple< Long<0>, Long<1>, Long<9> >
        >::Value,
        "Tuple::SetLastItem< Long<9> >"
    );

    show("SetItem< 1, Long<9> >:   ");
    show<Tup3::SetItem<1, Long<9> > >();
    newline();
    static_assert(
        SameType<
            Tup3::SetItem<1, Long<9> >,
            Tuple< Long<0>, Long<9>, Long<2> >
        >::Value,
        "Tuple::SetItem<1, Long<9> >"
    );

    show("Append< Long<3>, Long<4> > :   ");
    show<Tup3::Append<Long<3>, Long<4> > >();
    newline();
    static_assert(
        SameType<
            Tup3::Append<Long<3>, Long<4> >,
            Tuple< Long<0>, Long<1>, Long<2>, Long<3>, Long<4> >
        >::Value,
        "Tuple::Append< Long<3>, Long<4> >"
    );

    show("Extend<Tup3> :   ");
    show<Tup3::Extend<Tup3> >();
    newline();
    static_assert(
        SameType<
            Tup3::Extend<Tup3>,
            Tuple< Long<0>, Long<1>, Long<2>, Long<0>, Long<1>, Long<2> >
        >::Value,
        "Tuple::Extend<Tup3>"
    );

    show("------------");
    newline();

    show(tuple(1, 2, 3, 4.0, 5.0, 6.0));
}
