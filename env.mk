#/bin/bash

#ifndef VERBOSE
#.SILENT:
#endif

NCE?=4 # Number of CEs per CR

ifeq ($(TARGET), fsim)
	CC  = $(REDEFINE_LLVMPATH)/clang
	CPPFLAGS = -D__FUNCTION_SIM -D__NUMCE__=$(NCE)
	LCC = $(REDEFINE_LLVMPATH)/clang
	LDLIBS = $(REDEFINE_LIB)/fsim/libredefine_fsim.so
	LDFLAGS = -O3
else
	CC = $(REDEFINE_LLVMPATH)/clang
	CPPFLAGS = -D__NUMCE__=$(NCE) -I${RISCV}/riscv32-unknown-elf/include
#INFO:-fno-builtin-memset may make code unsafe
#INFO:-ffreestanding "When you specify this option, the compiler will not assume the presence of compiler-specific libraries. 
#	It will only generate calls that appear in the source code. The compiler assumes that the standard library may not exist 
#	and program startup may not necessarily be at main." 
	CFLAGS = -target riscv32-unknown-elf -ffunction-sections -fdata-sections -Wuninitialized -fno-builtin -ffreestanding -march=rv32imf -mabi=ilp32f
	CFLAGS += -fno-addrsig
	LCC = $(REDEFINE_LLVMPATH)/ld.lld
	#INFO:--print-gc-sections : prints removed sections
	LDFLAGS = -gc-sections -print-gc-sections -Map=$(OUTNAME).map -L$(REDEFINE_LIB)/ldscript -L$(RISCV)/riscv32-unknown-elf/lib -T$(LINKFILE) -O3
endif
CPPFLAGS += -I$(REDEFINE_INCLUDE)
CFLAGS += -O3 -Wno-implicit-function-declaration

check-build:
ifndef REDEFINE_INCLUDE
	$(error REDEFINE_INCLUDE is not defined)
endif
ifndef REDEFINE_LIB
	$(error REDEFINE_LIB is not defined)
endif
ifndef REDEFINE_LLVMPATH
	$(error REDEFINE_LLVMPATH is not defined)
endif
ifeq ($(TARGET), isasim)
ifndef RISCV
	$(error RISCV environment variable not defined)
endif
endif
ifeq ($(TARGET), isasim)
	make $(OUTNAME).elf
else
	make $(OUTNAME).elf TARGET=fsim
endif 

.PHONY : clean
clean:
	-rm -f *.elf *.o *.map
