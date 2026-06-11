#/bin/bash
OUTNAME := $(notdir $(shell pwd))
DEBUG ?= 0
POSITION ?= 4

-include ../../redefine_sdk/mm-redefine/REDEFINE/baremetal_examples/env.mk

CPPFLAGS += -Wall -DDEBUG_CODE=$(DEBUG) -DPOSITION=$(POSITION)

commonObjs = rope.o

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

rope.o      : rope.c rope.h
main.o : main.c rope.h