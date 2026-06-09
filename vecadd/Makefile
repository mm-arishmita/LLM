#/bin/bash

OUTNAME := $(notdir $(shell pwd))

-include ../../redefine_sdk/mm-redefine/REDEFINE/baremetal_examples/env.mk

CPPFLAGS += -Wall
commonObjs = vec_add.o

ifneq ($(TARGET), isasim) 
 objects = main.o $(commonObjs)
else
 objects = $(commonObjs)
 LINKFILE = link.ld
endif
	
$(OUTNAME).elf : $(objects)
	$(LCC) $(LDFLAGS) -o $@ $^ $(LDLIBS) 

%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<


VecAdd.o : vec_add.c
main.o :  main.c
