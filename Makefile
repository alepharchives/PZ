CFLAGS=-O3 -Wall
SOURCES := src/sse2.c src/pz.c

all: pz

pz:
	$(CC) $(CFLAGS) -o pz $(SOURCES)

clean:
	rm -f pz
