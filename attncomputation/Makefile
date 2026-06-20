#/bin/bash

OUTNAME := $(notdir $(shell pwd))

DEBUG ?= 0

-include ../../redefine_sdk/mm-redefine/REDEFINE/baremetal_examples/env.mk



CPPFLAGS += -Wall -DDEBUG_CODE=$(DEBUG)
commonObjs = att.o

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

att.o : att.c
main.o :  main.c
