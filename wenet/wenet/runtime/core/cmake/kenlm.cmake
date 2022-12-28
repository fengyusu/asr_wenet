
# third_party: kenlm
# On how to build grpc, you may refer to https://github.com/grpc/grpc
# We recommend manually recursive clone the repo to avoid internet connection problem


FetchContent_Declare(kenlm
  URL "file://${CMAKE_CURRENT_SOURCE_DIR}/fc_base/kenlm.tar.xz"

)
FetchContent_MakeAvailable(kenlm)
include_directories(${kenlm_SOURCE_DIR})

