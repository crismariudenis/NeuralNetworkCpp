# the compiler: gcc for C program, define as g++ for C++
CC = g++
# compiler flags:
#  -g      adds debugging information to the executable file
#  -Wall   turns on most, but not all, compiler warnings
#  -Wextra turns on even more compiler warnings
CFLAGS  =  -std=c++17 -O2 -I include/ -L lib/
LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm -pthread



TARGET = test/gym

run: $(TARGET).cpp 
	$(CC) $(TARGET).cpp  -o $(TARGET) $(CFLAGS) -I include/ -L lib/ ${LIBS}
	./$(TARGET)
	 $(RM) $(TARGET)

clean:
	$(RM) *.exe

