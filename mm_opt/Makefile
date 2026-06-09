#/bin/bash

OUTNAME := $(notdir $(shell pwd))

-include ../../redefine_sdk/mm-redefine/REDEFINE/baremetal_examples/env.mk

COLS ?= 4
ROWS ?= 4

#CPPFLAGS += -Wall -pg
CPPFLAGS += -Wall
commonObjs = mm_opt.o
#commonObjs = mmSeq.o

#LDFLAGS += -pg

ifneq ($(TARGET), isasim) 
 objects = main.o $(commonObjs)
else
 objects = $(commonObjs)
 LINKFILE = link.ld
endif
	
$(OUTNAME).elf : $(objects)
	$(LCC) $(LDFLAGS) -o $@ $^ $(LDLIBS) 

# "-ffp-contract=fast" compiler flag to use fmadd for fmul follwed by fadd
%.o : %.c
	$(CC) -D__NUMCOL__=$(COLS) -D__NUMROW__=$(ROWS) -ffp-contract=fast $(CPPFLAGS) $(CFLAGS) -c -o $@ $<


mm_opt.o : mm_opt.c
main.o :  main.c
