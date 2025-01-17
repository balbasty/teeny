#!/bin/sh

c++ \
    -std=c++11 \
    -ferror-limit=1 \
    -ftemplate-backtrace-limit=0 \
    -I../include \
    -o test_meta_vector \
    test_meta_vector.cpp
