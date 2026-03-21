cmake_minimum_required(VERSION 3.15)
project(Clipper2 VERSION 2.0.1 LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

##########################################################################
##########################################################################

# CLIPPER2_HI_PRECISION: See GetIntersectPoint() in clipper.core.h 
option(BUILD_SHARED_LIBS "Build shared libs" OFF)
set(CLIPPER2_USINGZ "ON" CACHE STRING "Build Clipper2Z, either \"ON\" or \"OFF\" or \"ONLY\"")
# CLIPPER2_MAX_DECIMAL_PRECISION: maximum decimal precision when scaling PathsD to Paths64.
# Caution: excessive scaling will increase the likelihood of integer overflow errors.
# See https://www.angusj.com/clipper2/Docs/Overview.htm#coord_range for more info.
set(CLIPPER2_MAX_DECIMAL_PRECISION 8 CACHE STRING "Maximum decimal precision range")

##########################################################################
##########################################################################

include(GNUInstallDirs)
set(CLIPPER2_INC_FOLDER ${PROJECT_SOURCE_DIR}/include/clipper2)

configure_file(clipper.version.in
  ${CLIPPER2_INC_FOLDER}/clipper.version.h NEWLINE_STYLE UNIX)

set(CLIPPER2_INC
  ${CLIPPER2_INC_FOLDER}/clipper.h
  ${CLIPPER2_INC_FOLDER}/clipper.version.h
  ${CLIPPER2_INC_FOLDER}/clipper.core.h
  ${CLIPPER2_INC_FOLDER}/clipper.engine.h
  ${CLIPPER2_INC_FOLDER}/clipper.export.h
  ${CLIPPER2_INC_FOLDER}/clipper.minkowski.h
  ${CLIPPER2_INC_FOLDER}/clipper.offset.h
  ${CLIPPER2_INC_FOLDER}/clipper.rectclip.h
  ${CLIPPER2_INC_FOLDER}/clipper.triangulation.h
)

set(CLIPPER2_SRC
  src/clipper.engine.cpp
  src/clipper.offset.cpp
  src/clipper.rectclip.cpp
  src/clipper.triangulation.cpp
)

set(CLIPPER2_LIBS "") # one or both of Clipper2/Clipper2Z

# primary Clipper2 library
if (NOT (CLIPPER2_USINGZ STREQUAL "ONLY"))
  list(APPEND CLIPPER2_LIBS Clipper2)
  add_library(Clipper2 ${CLIPPER2_INC} ${CLIPPER2_SRC})

  target_compile_definitions(
    Clipper2 PUBLIC
      CLIPPER2_MAX_DECIMAL_PRECISION=${CLIPPER2_MAX_DECIMAL_PRECISION}
      $<$<BOOL:${CLIPPER2_HI_PRECISION}>:CLIPPER2_HI_PRECISION>
  )

  target_include_directories(
    Clipper2 PUBLIC 
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )


  target_compile_options(Clipper2 PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(Clipper2 PUBLIC -lm)
endif()

set_target_properties(${CLIPPER2_LIBS} PROPERTIES FOLDER Libraries
                                         VERSION ${PROJECT_VERSION}
                                         SOVERSION ${PROJECT_VERSION_MAJOR}
                                         PUBLIC_HEADER "${CLIPPER2_INC}"
)

set(CLIPPER2_PCFILES "")
foreach(lib ${CLIPPER2_LIBS})
  set(pc "${CMAKE_CURRENT_BINARY_DIR}/${lib}.pc")
  list(APPEND CLIPPER2_PCFILES ${pc})
  if (lib STREQUAL "Clipper2Z")
    set(PCFILE_LIB_SUFFIX "Z")
  else()
    set(PCFILE_LIB_SUFFIX "")
  endif()
  configure_file(Clipper2.pc.cmakein "${pc}" @ONLY)
endforeach()

install(TARGETS ${CLIPPER2_LIBS}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/clipper2
)
install(FILES ${CLIPPER2_PCFILES} DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)


# create package config file
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

write_basic_package_version_file("${PROJECT_BINARY_DIR}/Clipper2ConfigVersion.cmake" COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    "Clipper2Config.cmake.in" 
    "${PROJECT_BINARY_DIR}/Clipper2Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clipper2
)

export(TARGETS ${CLIPPER2_LIBS} FILE "${PROJECT_BINARY_DIR}/Clipper2Targets.cmake")

# installation
install(TARGETS ${CLIPPER2_LIBS} EXPORT Clipper2-targets
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/clipper2
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    FILES 
        ${PROJECT_BINARY_DIR}/Clipper2Config.cmake 
        ${PROJECT_BINARY_DIR}/Clipper2ConfigVersion.cmake 
    DESTINATION 
        ${CMAKE_INSTALL_LIBDIR}/cmake/clipper2
)

install(
    EXPORT 
        Clipper2-targets
    NAMESPACE 
        Clipper2::
    FILE 
        Clipper2Targets.cmake 
    DESTINATION 
        ${CMAKE_INSTALL_LIBDIR}/cmake/clipper2
)
