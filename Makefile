CC=xlc++
CC_FLAGS=-fopenmp -O3 -std=c++11 -qtgtarch=sm_60 -qoffload 

SOURCES=lax_wave.cpp
OBJECTS=lax_wave.o

TARGET=OMP_LAX

%.o: %.cpp
	$(CC) $(CC_FLAGS) -c $^

$(TARGET): $(OBJECTS)
	$(CC) $(CC_FLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o $(TARGET) *~
