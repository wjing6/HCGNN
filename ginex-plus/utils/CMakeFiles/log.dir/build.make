# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /data00/home/liuyibo/local/miniconda3/envs/cuda-3.9/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /data00/home/liuyibo/local/miniconda3/envs/cuda-3.9/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils

# Include any dependencies generated for this target.
include CMakeFiles/log.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/log.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/log.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/log.dir/flags.make

CMakeFiles/log.dir/log.o: CMakeFiles/log.dir/flags.make
CMakeFiles/log.dir/log.o: log.cpp
CMakeFiles/log.dir/log.o: CMakeFiles/log.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/log.dir/log.o"
	/home/tiger/liuyibo/local/miniconda3/envs/cuda-3.9/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/log.dir/log.o -MF CMakeFiles/log.dir/log.o.d -o CMakeFiles/log.dir/log.o -c /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils/log.cpp

CMakeFiles/log.dir/log.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/log.dir/log.i"
	/home/tiger/liuyibo/local/miniconda3/envs/cuda-3.9/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils/log.cpp > CMakeFiles/log.dir/log.i

CMakeFiles/log.dir/log.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/log.dir/log.s"
	/home/tiger/liuyibo/local/miniconda3/envs/cuda-3.9/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils/log.cpp -o CMakeFiles/log.dir/log.s

# Object files for target log
log_OBJECTS = \
"CMakeFiles/log.dir/log.o"

# External object files for target log
log_EXTERNAL_OBJECTS =

liblog.so: CMakeFiles/log.dir/log.o
liblog.so: CMakeFiles/log.dir/build.make
liblog.so: /home/tiger/liuyibo/local/miniconda3/envs/cuda-3.9/lib/libgomp.so
liblog.so: /home/tiger/liuyibo/local/miniconda3/envs/cuda-3.9/x86_64-conda-linux-gnu/sysroot/usr/lib/libpthread.so
liblog.so: CMakeFiles/log.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library liblog.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/log.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/log.dir/build: liblog.so
.PHONY : CMakeFiles/log.dir/build

CMakeFiles/log.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/log.dir/cmake_clean.cmake
.PHONY : CMakeFiles/log.dir/clean

CMakeFiles/log.dir/depend:
	cd /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils /home/tiger/liuyibo/local/torch-quiver/ginex-plus/utils/CMakeFiles/log.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/log.dir/depend

