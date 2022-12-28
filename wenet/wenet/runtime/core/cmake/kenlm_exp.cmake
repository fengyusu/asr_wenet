
# third_party: kenlm
# On how to build grpc, you may refer to https://github.com/grpc/grpc
# We recommend manually recursive clone the repo to avoid internet connection problem

include(ExternalProject)

set(KENLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../fc_base/kenlm)

set(KENLM_GIT_URL      https://github.com/fengyusu/kenlm.git)  # 指定git仓库地址
set(KENLM_CONFIGURE    cd ${KENLM_ROOT} && mkdir build && mkdir install && cd build && cmake -D CMAKE_INSTALL_PREFIX=${KENLM_ROOT}/install  ..)  # 指定配置指令（注意此处修改了安装目录，否则默认情况下回安装到系统目录）
set(KENLM_MAKE         cd ${KENLM_ROOT}/build && make)  # 指定编译指令（需要覆盖默认指令，进入我们指定的SPDLOG_ROOT目录下）
set(KENLM_INSTALL      cd ${CMAKE_CURRENT_SOURCE_DIR})  # 指定安装指令（需要覆盖默认指令，进入我们指定的SPDLOG_ROOT目录下）

ExternalProject_Add(kenlm
        PREFIX            ${KENLM_ROOT}
        GIT_REPOSITORY    ${KENLM_GIT_URL}
        CONFIGURE_COMMAND ${KENLM_CONFIGURE}
        BUILD_COMMAND     ${KENLM_MAKE}
        INSTALL_COMMAND   ${KENLM_INSTALL}
)

# 指定编译好的静态库文件的路径
set(KENLM_LIB       ${KENLM_ROOT}/build/lib/libkenlm_builder.so
                    ${KENLM_ROOT}/build/lib/libkenlm_filter.so
                    ${KENLM_ROOT}/build/lib/libkenlm_util.so
                    ${KENLM_ROOT}/build/lib/libkenlm.so)
# 指定头文件所在的目录
set(KENLM_INCLUDE_DIR   ${KENLM_ROOT})