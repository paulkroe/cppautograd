# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/paul/Desktop/cpptrans/vanilla

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/paul/Desktop/cpptrans/vanilla/build

# Include any dependencies generated for this target.
include CMakeFiles/cppgrad.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cppgrad.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cppgrad.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cppgrad.dir/flags.make

CMakeFiles/cppgrad.dir/cppgrad.cpp.o: CMakeFiles/cppgrad.dir/flags.make
CMakeFiles/cppgrad.dir/cppgrad.cpp.o: /home/paul/Desktop/cpptrans/vanilla/cppgrad.cpp
CMakeFiles/cppgrad.dir/cppgrad.cpp.o: CMakeFiles/cppgrad.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/paul/Desktop/cpptrans/vanilla/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cppgrad.dir/cppgrad.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cppgrad.dir/cppgrad.cpp.o -MF CMakeFiles/cppgrad.dir/cppgrad.cpp.o.d -o CMakeFiles/cppgrad.dir/cppgrad.cpp.o -c /home/paul/Desktop/cpptrans/vanilla/cppgrad.cpp

CMakeFiles/cppgrad.dir/cppgrad.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cppgrad.dir/cppgrad.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/paul/Desktop/cpptrans/vanilla/cppgrad.cpp > CMakeFiles/cppgrad.dir/cppgrad.cpp.i

CMakeFiles/cppgrad.dir/cppgrad.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cppgrad.dir/cppgrad.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/paul/Desktop/cpptrans/vanilla/cppgrad.cpp -o CMakeFiles/cppgrad.dir/cppgrad.cpp.s

# Object files for target cppgrad
cppgrad_OBJECTS = \
"CMakeFiles/cppgrad.dir/cppgrad.cpp.o"

# External object files for target cppgrad
cppgrad_EXTERNAL_OBJECTS =

libcppgrad.a: CMakeFiles/cppgrad.dir/cppgrad.cpp.o
libcppgrad.a: CMakeFiles/cppgrad.dir/build.make
libcppgrad.a: CMakeFiles/cppgrad.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/paul/Desktop/cpptrans/vanilla/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcppgrad.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cppgrad.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cppgrad.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cppgrad.dir/build: libcppgrad.a
.PHONY : CMakeFiles/cppgrad.dir/build

CMakeFiles/cppgrad.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cppgrad.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cppgrad.dir/clean

CMakeFiles/cppgrad.dir/depend:
	cd /home/paul/Desktop/cpptrans/vanilla/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/paul/Desktop/cpptrans/vanilla /home/paul/Desktop/cpptrans/vanilla /home/paul/Desktop/cpptrans/vanilla/build /home/paul/Desktop/cpptrans/vanilla/build /home/paul/Desktop/cpptrans/vanilla/build/CMakeFiles/cppgrad.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cppgrad.dir/depend

