#  -g      adds debugging information to the executable file
#  -Wall   turns on most, but not all, compiler warnings
#  -Wextra turns on even more compiler warnings
CC = g++
CFLAGS  = -g -std=c++17 -O2 -I include/ -L lib/
LIBS = -lraylib -pthread

FILE = adder
TARGET = src/$(FILE)
EXECUTABLE=build/$(FILE)

run: build
	./$(EXECUTABLE)
	rm $(EXECUTABLE)
	
build: $(EXECUTABLE)

$(EXECUTABLE): $(TARGET).cpp 
	$(CC) $(TARGET).cpp -o $(EXECUTABLE) $(CFLAGS) $(LIBS)



clean:
	$(RM) build/*	

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib