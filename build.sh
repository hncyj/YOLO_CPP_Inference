#!/bin/bash

BUILD_TYPE=${1:-Debug}

if [[ -d "build" ]]; then
    rm -rf build
fi

set -e
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" .. && cmake --build .