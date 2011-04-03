CFLAGS=-O3 -Wall
TFLAGS=-DTEST
SRC := src/sse2.c
PSRC := $(SRC) src/pz.c
TSRC := $(SRC) src/test.c

all: pz test

pz: $(SRC)
	@echo "making pz"
	$(CC) $(CFLAGS) -o pz $(SRC) $(PSRC)

test: $(TSRC)
	@echo "making test"
	$(CC) $(CFLAGS) $(TFLAGS) -o test $(TSRC)

clean:
	rm -f pz test
