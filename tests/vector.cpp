#include "../include/miniten/meta/vector.h"
#include "../include/miniten/meta/show.h"


using namespace miniten;
using namespace miniten::meta;

int main()
{
    using Vec0 = Long<>;
    show<Vec0>();
    newline();

    using Vec1 = Long<0>;
    show<Vec1>();
    newline();

    using Vec2 = Long<0, 1>;
    show<Vec2>();
    newline();

    using Vec3 = Long<0, 1, 2>;
    show<Vec3>();
    newline();

    show("DelFirstItem: ");
    show<Vec3::DelFirstItem>();
    newline();
    static_assert(SameType<Vec3::DelFirstItem, Long<1, 2> >::Value,
                  "Vector::DelFirstItem");

    show("DelLastItem:  ");
    show<Vec3::DelLastItem>();
    newline();
    static_assert(SameType<Vec3::DelLastItem, Long<0, 1> >::Value,
                  "Vector::DelLastItem");

    show("Del<1>:       ");
    show<Vec3::Del<1> >();
    newline();
    static_assert(SameType<Vec3::Del<1> , Long<0, 2> >::Value,
                  "Vector::Del<1> ");

    show("Del<-2>:      ");
    show<Vec3::Del<-2> >();
    newline();
    static_assert(SameType<Vec3::Del<-2> , Long<0, 2> >::Value,
                  "Vector::Del<-2> ");

    show("GetFirstItem: ");
    show(Vec3::GetFirstItem);
    newline();
    static_assert(Vec3::GetFirstItem == 0,
                  "Vector::GetFirstItem ");

    show("GetLastItem:  ");
    show(Vec3::GetLastItem);
    newline();
    static_assert(Vec3::GetLastItem == 2,
                  "Vector::GetLastItem ");

    show("Get<1>:       ");
    show<Vec3::Get<1> >();
    newline();
    static_assert(SameType<Vec3::Get<1> , Long<1> >::Value,
                  "Vector::Get<1> ");

    show("Get<-2>:      ");
    show<Vec3::Get<-2> >();
    newline();
    static_assert(SameType<Vec3::Get<-2> , Long<1> >::Value,
                  "Vector::Get<-2> ");

    show("Get<1, 2>:    ");
    show<Vec3::Get<1, 2> >();
    newline();
    static_assert(SameType<Vec3::Get<1, 2> , Long<1, 2> >::Value,
                  "Vector::Get<1, 2> ");

    show("Slice<0, 2>:  ");
    show< Vec3::Slice<0, 2> >();
    newline();
    static_assert(SameType<Vec3::Slice<0, 2>, Long<0, 1> >::Value,
                  "Vector::Slice<0, 2>");

    show("Slice<0, 0>:  ");
    show< Vec3::Slice<0, 0> >();
    newline();
    static_assert(SameType<Vec3::Slice<0, 0>, Long<> >::Value,
                  "Vector::Slice<0, 0>");

    show("Slice<2, 2>:  ");
    show< Vec3::Slice<2, 2> >();
    newline();
    static_assert(SameType<Vec3::Slice<2, 2>, Long<> >::Value,
                  "Vector::Slice<2, 2>");

    show("SetFirstItem<9>: ");
    show<Vec3::SetFirstItem<9> >();
    newline();
    static_assert(SameType<Vec3::SetFirstItem<9> , Long<9, 1, 2> >::Value,
                  "Vector::SetFirstItem<9> ");

    show("SetLastItem<9>:  ");
    show<Vec3::SetLastItem<9> >();
    newline();
    static_assert(SameType<Vec3::SetLastItem<9> , Long<0, 1, 9> >::Value,
                  "Vector::SetLastItem<9>");

    show("SetItem<1, 9>:   ");
    show<Vec3::SetItem<1, 9> >();
    newline();
    static_assert(SameType<Vec3::SetItem<1, 9> , Long<0, 9, 2> >::Value,
                  "Vector::SetItem<1, 9>");

    show("Append<3, 4> :   ");
    show<Vec3::Append<3, 4> >();
    newline();

    show("Extend<Vec3> :   ");
    show<Vec3::Extend<Vec3> >();
    newline();
}
