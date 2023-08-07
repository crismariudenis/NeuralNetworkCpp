#  -g      adds debugging information to the executable file
#  -Wall   turns on most, but not all, compiler warnings
#  -Wextra turns on even more compiler warnings
CC = g++
CFLAGS  =  -std=c++17 -O2 -I include/ -L lib/
LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm -pthread

FILE = gym
TARGET = src/$(FILE)
EXECUTABLE=build/$(FILE)

run: build
	./$(EXECUTABLE)
	
build: $(EXECUTABLE)

$(EXECUTABLE): $(TARGET).cpp 
	$(CC) $(TARGET).cpp -o $(EXECUTABLE) $(CFLAGS) $(LIBS)



clean:
	$(RM) build/*.exe	

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib