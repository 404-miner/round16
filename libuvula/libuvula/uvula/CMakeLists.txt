set(UVULA_SRC
    src/xatlas.cpp
    src/unwrap.cpp
    src/project.cpp
    src/Vector3F.cpp
    src/Vector2F.cpp
    src/Matrix33F.cpp
    src/Matrix44F.cpp
    src/Triangle3F.cpp
    src/Point3F.cpp
    src/geometry_utils.cpp
)
add_library(libuvula STATIC ${UVULA_SRC})
set_target_properties(libuvula PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(libuvula
        PUBLIC
        spdlog::spdlog
        range-v3::range-v3
        Clipper2
)

target_include_directories(libuvula
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libuvula/uvula/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(libuvula
        PUBLIC
        UVULA_VERSION="${PYUVULA_VERSION}"
)

target_link_libraries(libuvula PRIVATE Threads::Threads)