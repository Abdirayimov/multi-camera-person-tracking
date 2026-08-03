include(FetchContent)

# Pinned rather than tracking main: an unannounced GoogleTest change should
# never be able to turn this repository's CI red.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# GoogleTest is third-party; this project's warning set is not its contract.
get_directory_property(MCT_WARNINGS COMPILE_OPTIONS)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "")
FetchContent_MakeAvailable(googletest)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${MCT_WARNINGS}")

include(GoogleTest)

add_executable(mc_tracking_tests
    test_assignment.cpp
    test_kalman.cpp
    test_trackers.cpp
    test_reid_gallery.cpp
    test_identity_matcher.cpp
    test_config.cpp
)
target_link_libraries(mc_tracking_tests PRIVATE mc_tracking_core GTest::gtest_main)
target_compile_definitions(mc_tracking_tests
    PRIVATE MCT_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

gtest_discover_tests(mc_tracking_tests)
