
#/bin/bash
OUTNAME := $(notdir $(shell pwd))
DEBUG ?= 0
SEQ_LEN ?= 20

-include ../../redefine_sdk/mm-redefine/REDEFINE/baremetal_examples/env.mk

CPPFLAGS += -Wall -DDEBUG_CODE=$(DEBUG) -DSEQ_LEN=$(SEQ_LEN)

commonObjs = wsum.o

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

wsum.o     : wsum.c wsum.h
main.o : main.c wsum.h
