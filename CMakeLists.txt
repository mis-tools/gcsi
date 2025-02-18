cmake_minimum_required(VERSION 2.8.12)
project(gcsi)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

FIND_PACKAGE( Boost 1.40 )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

add_executable(gcsi src/gcsi.cxx)
target_link_libraries(gcsi ${ITK_LIBRARIES})

INSTALL(TARGETS gcsi DESTINATION bin/)
