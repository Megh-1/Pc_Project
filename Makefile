CC      ?= gcc
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra
LDLIBS  := -lm

all: bs_fdm

bs_fdm: bs_fdm.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

# Confirms whether the hot loops were auto-vectorised. Referenced in §6.4 of
# the report -- run this before claiming any AVX speedup in Milestone 3.
vecreport: bs_fdm.c
	$(CC) $(CFLAGS) -fopt-info-vec-optimized -c $< -o /dev/null 2>&1 | grep -i stencil || true

# Honest scalar reference point, for isolating the compiler's contribution.
bs_fdm_scalar: bs_fdm.c
	$(CC) -O2 -fno-tree-vectorize -std=c11 $< -o $@ $(LDLIBS)

results: bs_fdm
	@mkdir -p results
	./bs_fdm price    > results/price.txt
	./bs_fdm converge > results/converge.txt
	./bs_fdm american > results/american.txt
	./bs_fdm bench    > results/bench.txt
	./bs_fdm 2d       > results/2d.txt
	@echo "raw output written to results/"

figures: results
	python3 make_figures.py

clean:
	rm -f bs_fdm bs_fdm_scalar *.o *.png
	rm -rf results

.PHONY: all vecreport results figures clean
