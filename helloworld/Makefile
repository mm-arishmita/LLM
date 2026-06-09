#/bin/bash

OUTNAME := $(notdir $(shell pwd))

-include ../../redefine_sdk/mm-redefine/REDEFINE/baremetal_examples/env.mk

COLS ?= 2
ROWS ?= 2
CPPFLAGS += -Wall

commonObjs = hello.o

ifneq ($(TARGET), isasim) 
 objects = main.o $(commonObjs)
else
 objects = $(commonObjs)
 LINKFILE = link.ld
endif
	
$(OUTNAME).elf : $(objects)
	$(LCC) $(LDFLAGS) -o $@ $^ $(LDLIBS) 

%.o : %.c
	$(CC) $(CPPFLAGS) -D__NUMCOL__=$(COLS) -D__NUMROW__=$(ROWS) $(CFLAGS) -c -o $@ $<


hello.o : hello.c
main.o :  main.c

.PHONY : clean 
clean : 
	-rm -f *.elf *.o ./bin/*