# Overview
A fork from Sweeney's [TheiaSfM](https://github.com/sweeneychris/TheiaSfM) with bug fixes, and updates.  I chose not to use GitHub's fork feature because the upstream repository has been inactive for a long time.

# TODO
- [x] Fix bugs and simplify the compilation process on the Windows platform. Use vcpkg and replace OpenImageIO with FreeImage.
- [ ] Refactor the code structure.
# Installation
```
cd path/to/theiaX
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
```
Alternatively, you can use a CMake preset by creating a CMakeUserPresets.json file. Here’s an example:
```json
{
  "version": 2,
  "configurePresets": [
    {
      "name": "default",
      "generator": "Visual Studio 17 2022",
      "binaryDir": "${sourceDir}/build",
      "environment": {
        "VCPKG_ROOT": "path/to/vcpkg/"
      },
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
        "CMAKE_CXX_STANDARD": "17",
        "CMAKE_CXX_STANDARD_REQUIRED": "ON",
        "CMAKE_GENERATOR_PLATFORM": "x64",
        "CMAKE_CXX_FLAGS": "/utf-8 /EHsc"
      }
    }
  ]
}
```
Then, configure and build using the preset: 
```
cmake --preset=default
cmake --build build --config Release --target ALL_BUILD -j12
```
To run all tests, use:
```
ctest -C Release
```
Original TheiaSfM README
---

Copyright 2015-2016 Chris Sweeney (sweeney.chris.m@gmail.com)
UC Santa Barbara

What is this library?
---------------------

Theia is an end-to-end structure-from-motion library that was created by Chris
Sweeney. It is designed to be very efficient, scalable, and accurate. All
steps of the pipeline are designed to be modular so that code is easy to read
and easy to extend.

Please see the Theia website for detailed information, including instructions
for building Theia and full documentation of the library. The website is
currently located at http://www.theia-sfm.org

Contact Information
-------------------

Questions, comments, and bug reports can be sent to the Theia mailing list:
theia-vision-library@googlegroups.com

Citing this library
-------------------

If you are using this library for academic research or publications we ask that
you please cite this library as:

    @misc{theia-manual,
      author = {Chris Sweeney},
      title = {Theia Multiview Geometry Library: Tutorial \& Reference},
      howpublished = "\url{http://theia-sfm.org}",
    }
