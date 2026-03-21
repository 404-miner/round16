add_subdirectory(${PROJECT_SOURCE_DIR}/libuvula/Clipper2Lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/libuvula/uvula)

set(CLIPPER2_INC_FOLDER ${PROJECT_SOURCE_DIR}/libuvula/Clipper2Lib/include/clipper2)
configure_file(${PROJECT_SOURCE_DIR}/libuvula/Clipper2Lib/clipper.version.in ${CLIPPER2_INC_FOLDER}/clipper.version.h NEWLINE_STYLE UNIX)

# --- Setup Python bindings ---
if (WITH_PYTHON_BINDINGS)
    add_subdirectory(pyUvula)
endif ()
