# GPT-OSS 20B C Inference Engine — Build Targets
# Usage: make win    (Windows with MinGW)
#        make linux  (Linux with GCC)
#        make nomp   (No OpenMP, single-threaded)

CC = gcc
CFLAGS = -O3 -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -lm

# Windows MinGW path (update if different)
CC_WIN = "C:/Users/naman/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/gcc.exe"

# Default target
all: win

# Windows (MinGW, with OpenMP)
win:
	$(CC_WIN) $(CFLAGS) -fopenmp -o run_gptoss.exe run_gptoss.c $(LDFLAGS) -lshell32

# Linux (with OpenMP)
linux:
	$(CC) $(CFLAGS) -fopenmp -o run_gptoss run_gptoss.c $(LDFLAGS)

# No OpenMP (single-threaded, any platform)
nomp:
	$(CC) $(CFLAGS) -o run_gptoss run_gptoss.c $(LDFLAGS)

# Windows no OpenMP
win-nomp:
	$(CC_WIN) $(CFLAGS) -o run_gptoss.exe run_gptoss.c $(LDFLAGS) -lshell32

# Debug build
debug:
	$(CC) -g -O0 -Wall -Wextra -fsanitize=address -o run_gptoss_dbg run_gptoss.c $(LDFLAGS)

clean:
	rm -f run_gptoss run_gptoss.exe run_gptoss_dbg

.PHONY: all win linux nomp win-nomp debug clean
