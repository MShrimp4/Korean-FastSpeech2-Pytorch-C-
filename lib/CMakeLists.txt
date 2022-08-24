cmake_minimum_required(VERSION 3.9)
project(fastspeech VERSION 1.0.0 DESCRIPTION "Kor-FastSpeech w/ libtorch")
include(GNUInstallDirs)#find_package(GNUInstallDirs REQUIRED)
include(FindPkgConfig)#find_package(PkgConfig REQUIRED)
add_library(fastspeech SHARED fastspeech.cpp misc.cpp utils.cpp)
set_target_properties(fastspeech PROPERTIES
	VERSION ${PROJECT_VERSION}
	SOVERSION 1
	PUBLIC_HEADER fastspeech.h)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(Torch REQUIRED
             g2pk REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
target_link_libraries(fastspeech g2pk "${TORCH_LIBRARIES}")
target_include_directories(fastspeech PRIVATE .)

configure_file(fastspeech.pc.in fastspeech.pc @ONLY)
#configure_file(fastspeech_conf.h.in fastspeech_conf.h)

install(TARGETS fastspeech
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/fastspeech.pc
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
