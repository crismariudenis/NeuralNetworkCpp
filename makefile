#  -g      adds debugging information to the executable file
#  -Wall   turns on most, but not all, compiler warnings
#  -Wextra turns on even more compiler warnings
CC = g++
CFLAGS  =-g -std=c++17 -O3 -I include/ -L lib/
SHELL := $(shell if command -v fish > /dev/null 2>&1; then echo /usr/bin/fish; else echo /bin/bash; fi)

ifeq ($(OS),Windows_NT)
    LIBS =-lraylib -lgdi32 -lwinmm -pthread
else
    LIBS =-lraylib -pthread
endif

# Change this to the name of the file you want to compile
FILE = adder
TARGET = demos/$(FILE)
EXECUTABLE=build/$(FILE)

run: build
	./$(EXECUTABLE)
	rm $(EXECUTABLE)

time: build_with_file
	time ./$(EXECUTABLE)
	rm $(EXECUTABLE)
    
build: $(EXECUTABLE)

build_with_file: CFLAGS += -DTIME=1
build_with_file: build

$(EXECUTABLE): $(TARGET).cpp 
	$(CC) $(TARGET).cpp -o $(EXECUTABLE) $(CFLAGS) $(LIBS)

clean:
	$(RM) build/*    

build_all: $(patsubst demos/%.cpp, build/%, $(wildcard demos/*.cpp))

build/%: demos/%.cpp
	$(CC) $< -o $@ $(CFLAGS) $(LIBS)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib