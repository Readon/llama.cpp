cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 50     == Maxwell, lowest CUDA 12 standard
        # 60     == P100, FP16 CUDA intrinsics
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        # 80     == Ampere, asynchronous data loading, faster tensor core instructions
        # 86     == RTX 3000, needs CUDA v11.1
        # 89     == RTX 4000, needs CUDA v11.8
        #
        # XX-virtual == compile CUDA code as PTX, do JIT compilation to binary code on first run
        # XX-real    == compile CUDA code as device code for this specific architecture
        # no suffix  == compile as both PTX and device code
        #
        # The default behavior for a non-native is to build virtual architectures as needed to cover all features needed
        #     for best performance and to also build real architectures for the most commonly used GPUs.
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        elseif(GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                set(CMAKE_CUDA_ARCHITECTURES "60-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "60-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real")
            endif()
        else()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                set(CMAKE_CUDA_ARCHITECTURES "50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real")
            endif()
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    enable_language(CUDA)

    file(GLOB   GGML_HEADERS_CUDA "*.cuh")
    list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")

    file(GLOB   GGML_SOURCES_CUDA "*.cu")
    file(GLOB   SRCS "template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        file(GLOB   SRCS "template-instances/fattn-vec*q4_0-q4_0.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB   SRCS "template-instances/fattn-vec*q8_0-q8_0.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB   SRCS "template-instances/fattn-vec*f16-f16.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
    endif()

    ggml_add_backend_library(ggml-cuda
                             ${GGML_HEADERS_CUDA}
                             ${GGML_SOURCES_CUDA}
                            )

    add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

    # Check for NCCL with multiple detection methods (only if enabled)
    set(NCCL_FOUND FALSE)

    if (NOT GGML_CUDA_NCCL)
        message(STATUS "NCCL detection disabled via GGML_CUDA_NCCL=OFF")
    else()

    # Support environment variables for NCCL detection
    if (DEFINED ENV{NCCL_ROOT})
        set(NCCL_ROOT $ENV{NCCL_ROOT})
    endif()
    if (DEFINED ENV{NCCL_DIR})
        set(NCCL_ROOT $ENV{NCCL_DIR})
    endif()

    # Method 1: Try standard CMake package detection
    find_package(NCCL QUIET)
    if (NCCL_FOUND)
        message(STATUS "NCCL found via CMake package")
        set(NCCL_LIBRARIES NCCL::nccl)
    else()
        # Method 2: Manual detection using find_path and find_library
        # Look for NCCL header
        find_path(NCCL_INCLUDE_DIR
            NAMES nccl.h
            HINTS
                ${NCCL_ROOT}/include
                ${CUDA_TOOLKIT_ROOT_DIR}/include
                $ENV{CUDA_HOME}/include
                /usr/include
                /usr/local/include
                /usr/local/cuda/include
                /opt/nccl/include
                /opt/cuda/include
                # Common conda/mamba paths
                $ENV{CONDA_PREFIX}/include
                $ENV{PREFIX}/include
                # Common module system paths
                /apps/nccl/include
                /sw/nccl/include
            PATH_SUFFIXES nccl cuda/include
        )

        # Look for NCCL library
        find_library(NCCL_LIBRARY
            NAMES nccl libnccl
            HINTS
                ${NCCL_ROOT}/lib
                ${NCCL_ROOT}/lib64
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                $ENV{CUDA_HOME}/lib64
                /usr/lib
                /usr/lib64
                /usr/local/lib
                /usr/local/lib64
                /usr/local/cuda/lib64
                /opt/nccl/lib
                /opt/nccl/lib64
                /opt/cuda/lib64
                # Common conda/mamba paths
                $ENV{CONDA_PREFIX}/lib
                $ENV{PREFIX}/lib
                # Common module system paths
                /apps/nccl/lib
                /apps/nccl/lib64
                /sw/nccl/lib
                /sw/nccl/lib64
            PATH_SUFFIXES nccl cuda/lib64
        )

        # Check if both header and library were found
        if (NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
            set(NCCL_FOUND TRUE)
            message(STATUS "NCCL found manually: ${NCCL_LIBRARY}")
            message(STATUS "NCCL include dir: ${NCCL_INCLUDE_DIR}")
            set(NCCL_LIBRARIES ${NCCL_LIBRARY})
            set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
        endif()
    endif()
    endif() # End of GGML_CUDA_NCCL check

    if (NCCL_FOUND)
        message(STATUS "NCCL found, enabling tensor parallelism")
        add_compile_definitions(GGML_USE_NCCL)
        target_link_libraries(ggml-cuda PRIVATE ${NCCL_LIBRARIES})
        if (NCCL_INCLUDE_DIRS)
            target_include_directories(ggml-cuda PRIVATE ${NCCL_INCLUDE_DIRS})
        endif()
    else()
        if (GGML_CUDA_NCCL)
            message(WARNING "NCCL not found, tensor parallelism will be disabled")
            message(STATUS "To enable NCCL, install it or set NCCL_ROOT to the installation directory")
            message(STATUS "You can also disable this warning with -DGGML_CUDA_NCCL=OFF")
        else()
            message(STATUS "NCCL detection disabled, tensor parallelism will use basic mode")
        endif()
    endif()

    if (GGML_CUDA_GRAPHS)
        add_compile_definitions(GGML_CUDA_USE_GRAPHS)
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_VMM)
        add_compile_definitions(GGML_CUDA_NO_VMM)
    endif()

    if (NOT GGML_CUDA_FA)
        add_compile_definitions(GGML_CUDA_NO_FA)
    endif()

    if (GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
        add_compile_definitions(GGML_CUDA_F16)
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (GGML_STATIC)
        if (WIN32)
            # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
            target_link_libraries(ggml-cuda PRIVATE CUDA::cudart_static CUDA::cublas)
        else ()
            target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static)
        endif()
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    if (GGML_CUDA_NO_VMM)
        # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)
    endif()

    set(CUDA_CXX_FLAGS "")

    set(CUDA_FLAGS -use_fast_math -extended-lambda)

    if (GGML_CUDA_DEBUG)
        list(APPEND CUDA_FLAGS -lineinfo)
    endif()

    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        # Options are:
        # - none (not recommended)
        # - speed (nvcc's default)
        # - balance
        # - size
        list(APPEND CUDA_FLAGS -compress-mode=${GGML_CUDA_COMPRESSION_MODE})
    endif()

    if (GGML_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (GGML_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message(STATUS "CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    endif()

    list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()

    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
else()
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()
