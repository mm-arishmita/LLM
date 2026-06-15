# =============================================================================
# Makefile  —  WEIGHTED SUM COMPUTTAION
# =============================================================================

PROJ = $(notdir $(shell pwd))
##PROJ = att  

CC = gcc
CFLAGS = -std=c99 -Wall -DUNIX -g
LIBS   = -lm


# 32-bit vs 64-bit
PROC_TYPE = $(strip $(shell uname -m | grep 64))

# Mac vs Linux
OS     = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN = $(strip $(findstring DARWIN, $(OS)))

ifneq ($(DARWIN),)
    # ----- macOS -----
    CFLAGS += -DMAC
    LIBS   += -framework OpenCL
    ifeq ($(PROC_TYPE),)
        CFLAGS += -arch i386
    else
        CFLAGS += -arch x86_64
    endif
else
    # ----- Linux -----
    LIBS += -lOpenCL
    ifeq ($(PROC_TYPE),)
        CFLAGS += -m32
    else
        CFLAGS += -m64
    endif

    # --- OpenCL SDK include / lib paths ---
    # common.h and CL/opencl.h are found via these SDK paths,
    # exactly as in fib_fpga/Makefile. No need to hardcode paths
    # as long as one of these SDK env vars is set.

    ifdef AMDAPPSDKROOT
        # AMD APP SDK
        INC_DIRS = . $(AMDAPPSDKROOT)/include
        ifeq ($(PROC_TYPE),)
            LIB_DIRS = $(AMDAPPSDKROOT)/lib/x86
        else
            LIB_DIRS = $(AMDAPPSDKROOT)/lib/x86_64
        endif
    else
    ifdef NVSDKCOMPUTE_ROOT
        # Nvidia OpenCL SDK
        INC_DIRS = . $(NVSDKCOMPUTE_ROOT)/OpenCL/common/inc
    else
    ifdef REDEFINE_SDK_ROOT
        # Redefine SDK (set this if using Redefine's own OpenCL runtime)
        # export REDEFINE_SDK_ROOT=/path/to/redefine/sdk
        INC_DIRS = . $(REDEFINE_SDK_ROOT)/include
        LIB_DIRS = $(REDEFINE_SDK_ROOT)/lib
    endif
    endif
    endif
endif

#SOURCE FILES

SRCS = cltest.c common.c


#Build rules
# FPGA / RTL sim build  (rsim_hyperop device, no __FUNCTSIM__)
# Mirrors fib_fpga/Makefile "else" branch (TARGET != isasim)
$(PROJ).elf: $(SRCS)
ifneq ($(TARGET), isasim)
	$(CC) -D__FUNCTSIM $(CFLAGS) -o $@ $^ \
	    $(INC_DIRS:%=-I%) \
	    $(LIB_DIRS:%=-L%) \
	    $(LIBS)
else
	$(CC) $(CFLAGS) -o $@ $^ \
	    $(INC_DIRS:%=-I%) \
	    $(LIB_DIRS:%=-L%) \
	    $(LIBS)
endif

# -----------------------------------------------------------------------------
# Clean — remove binary and logs
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	-rm -f *.elf *.log
