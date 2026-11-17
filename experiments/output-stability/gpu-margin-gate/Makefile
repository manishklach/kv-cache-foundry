NVCC ?= nvcc
CUDA_ARCH ?= sm_90
NVCCFLAGS ?= -O3 -lineinfo -arch=$(CUDA_ARCH)

.PHONY: all run clean

all: margin_gate

margin_gate: margin_gate.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: margin_gate
	./margin_gate

clean:
	rm -f margin_gate
