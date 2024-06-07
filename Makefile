INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt


all: main

main: main.o
	$(CC) -O0 -g -I/usr/include/opencv4 -o main main.o  `pkg-config --libs opencv4` -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt


main.o: main.cpp
	$(CC) -O0 -g -I/usr/include/opencv4 -c main.cpp

clean:
	rm main main.o output*.avi output*.mp4

