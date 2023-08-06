# the compiler: gcc for C program, define as g++ for C++
CC = g++
# compiler flags:
#  -g      adds debugging information to the executable file
#  -Wall   turns on most, but not all, compiler warnings
#  -Wextra turns on even more compiler warnings
CFLAGS  =  -g -Wall -Wextra -std=c++17 -O2
LIBS = -lraylib -pthread



TARGET = test/gym

run: $(TARGET).cpp 
	$(CC) $(TARGET).cpp  -o $(TARGET) $(CFLAGS) ${LIBS}
	./$(TARGET)
	 $(RM) $(TARGET)

clean:
	$(RM) *.exe


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib