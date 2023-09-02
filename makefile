#  -g      adds debugging information to the executable file
#  -Wall   turns on most, but not all, compiler warnings
#  -Wextra turns on even more compiler warnings
CC = g++
CFLAGS  =-g -std=c++17 -O3 -I include/ -L lib/

ifeq ($(OS),Windows_NT)
    LIBS =-lraylib -lgdi32 -lwinmm -pthread
else
    LIBS =-lraylib -pthread
endif

# Change this to the name of the file you want to compile
FILE = xor
TARGET = demos/$(FILE)
EXECUTABLE=build/$(FILE)

run: build
	./$(EXECUTABLE)
	rm $(EXECUTABLE)
    
build: $(EXECUTABLE)

$(EXECUTABLE): $(TARGET).cpp 
	$(CC) $(TARGET).cpp -o $(EXECUTABLE) $(CFLAGS) $(LIBS)

clean:
	$(RM) build/*    

build_all: $(patsubst demos/%.cpp, build/%, $(wildcard demos/*.cpp))

build/%: demos/%.cpp
	$(CC) $< -o $@ $(CFLAGS) $(LIBS)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib