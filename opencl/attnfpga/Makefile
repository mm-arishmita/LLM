# =============================================================================
# Makefile  —  Attention Score Computation (FPGA via OpenCL)
#
# Converted from att/Makefile (FSim) following the same pattern as
# fib_fpga/Makefile (cltest.c) vs fib/Makefile (FSim).
#
# Usage:
#   make              → build for FPGA/RTL sim  (rsim_hyperop device)
#   make TARGET=isasim → build for functional sim (rfsim_hyperop device)
#   make clean        → remove build artifacts
# =============================================================================

PROJ = $(notdir $(shell pwd))
##PROJ = att          

# -----------------------------------------------------------------------------
# Compiler — hardcoded to gcc, same as fib_fpga/Makefile
# FSim Makefile used $(CC) from env.mk; FPGA build uses gcc directly.
# -----------------------------------------------------------------------------
CC = gcc

# -----------------------------------------------------------------------------
# Base CFLAGS — mirrors fib_fpga/Makefile base flags
# -std=c99   : C99 for HyperOps C compatibility
# -Wall      : all warnings (present in both FSim and FPGA Makefiles)
# -DUNIX     : Redefine runtime Unix flag (from fib_fpga/Makefile)
# -g         : debug symbols
# -lm        : math library (needed for sqrtf — present in your FSim Makefile)
# -----------------------------------------------------------------------------
CFLAGS = -std=c99 -Wall -DUNIX -g
LIBS   = -lm

# -----------------------------------------------------------------------------
# Platform detection — identical to fib_fpga/Makefile
# Detects: Mac vs Linux, 32-bit vs 64-bit, AMD vs Nvidia OpenCL SDK
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Source files
#
# FSim att/Makefile  : att.o (kernel) + main.o (host)  linked via LCC/env.mk
# FPGA att/Makefile  : att_main_opencl.c + ../../common.c  compiled via gcc
#
# common.c is compiled directly (not as a separate .o) — same as fib_fpga.
# att_fpga.c is the kernel source loaded at runtime via clCreateProgramWithSource
# — it is NOT compiled here, just needs to be present alongside the binary.
# -----------------------------------------------------------------------------
SRCS = cltest2.c common.c

# -----------------------------------------------------------------------------
# Build rules
# -----------------------------------------------------------------------------

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
