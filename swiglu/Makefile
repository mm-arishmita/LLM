#/bin/bash

OUTNAME := $(notdir $(shell pwd))

DEBUG ?= 0

-include ./../env.mk

CPPFLAGS += -Wall -DDEBUG_CODE=$(DEBUG)
commonObjs = kernel.o

ifneq ($(TARGET), isasim) 
	objects = main.o $(commonObjs)
else
	objects = $(commonObjs)
	LINKFILE = link.ld
endif

$(OUTNAME).elf : $(objects)
	$(LCC) $(LDFLAGS) -o $@ $^ $(LDLIBS) -lm

%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

kernel.o : kernel.c
main.o :  main.c
