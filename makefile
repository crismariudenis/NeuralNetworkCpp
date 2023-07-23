# the compiler: gcc for C program, define as g++ for C++
CC = g++
# compiler flags:
#  -g      adds debugging information to the executable file
#  -Wall   turns on most, but not all, compiler warnings
#  -Wextra turns on even more compiler warnings
CFLAGS  = -g -Wall -Wextra

# the build target executable:
TARGET = test/nn


build: $(TARGET).cpp 
	$(CC) $(TARGET).cpp  -o $(TARGET) $(CFLAGS)

run: $(TARGET)
	./$(TARGET) 
	$(RM) $(TARGET)
	

clean:
	$(RM) $(TARGET)

