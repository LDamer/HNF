.PHONY: all build run clean

CC = g++
CFLAGS = -g -O2 -std=c++11 -pthread -march=native


all: build
	@./hnf

build: main.cpp utility.cpp utility.h
	@$(CC) $(CFLAGS) main.cpp utility.cpp -o hnf -lntl -lgmp -lm

run:
	@./hnf

clean:
	@rm -f hnf
