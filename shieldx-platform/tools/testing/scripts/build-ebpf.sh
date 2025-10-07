#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." # repo root
SRC=pkg/sandbox/bpf/syscall_tracer.c
OUT=pkg/sandbox/bpf/syscall_tracer.o
if ! command -v clang >/dev/null; then
  echo "clang not found; please install clang/llvm to build eBPF object" >&2
  exit 1
fi
echo "Building $OUT from $SRC"
clang -O2 -g -target bpf -c "$SRC" -o "$OUT"
echo "Built $OUT"
