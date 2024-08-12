cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(real-time-sim)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(real-time-sim main.cpp)
target_link_libraries(real-time-sim torch)
set_property(TARGET real-time-sim PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET real-time-sim
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:real-time-sim>)
endif (MSVC)
