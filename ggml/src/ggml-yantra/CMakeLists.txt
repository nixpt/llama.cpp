# ggml-yantra (nixpt/yantra#211). Links yantra's C ABI (yantra-capi) as a
# prebuilt static library; cmake never runs cargo. Build it first:
#   cargo build -p yantra-capi            # in a yantra checkout
# then configure llama.cpp with
#   -DGGML_YANTRA=ON -DYANTRA_LIB=<target>/debug/libyantra.a
#   -DYANTRA_INCLUDE_DIR=<yantra>/include

if (NOT YANTRA_LIB OR NOT EXISTS "${YANTRA_LIB}")
    message(FATAL_ERROR "GGML_YANTRA: set YANTRA_LIB to yantra-capi's libyantra.a (got '${YANTRA_LIB}')")
endif()
if (NOT YANTRA_INCLUDE_DIR OR NOT EXISTS "${YANTRA_INCLUDE_DIR}/yantra.h")
    message(FATAL_ERROR "GGML_YANTRA: set YANTRA_INCLUDE_DIR to yantra's include/ (got '${YANTRA_INCLUDE_DIR}')")
endif()

ggml_add_backend_library(ggml-yantra
                         ggml-yantra.cpp
                        )

target_include_directories(ggml-yantra PRIVATE ${YANTRA_INCLUDE_DIR})
# libyantra.a is a Rust staticlib: it needs the C runtime's threads, dl and m.
target_link_libraries(ggml-yantra PRIVATE ${YANTRA_LIB} pthread dl m)
