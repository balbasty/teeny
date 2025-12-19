#!/bin/sh

c++ \
    -std=c++17 \
    -ferror-limit=1 \
    -ftemplate-backtrace-limit=0 \
    -I../include \
    -I../external/cccl/libcudacxx/include \
    -o test_statix_carray \
    test_statix_carray.cpp
