#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR="$ROOT_DIR/dist"
mkdir -p "$OUT_DIR"

echo "Building wch-client for Windows/macOS/Linux..."

pushd "$ROOT_DIR" >/dev/null

GOOS=windows GOARCH=amd64   go build -o "$OUT_DIR/wch-client-windows-amd64.exe" ./cmd/wch-client
GOOS=windows GOARCH=arm64   go build -o "$OUT_DIR/wch-client-windows-arm64.exe" ./cmd/wch-client
GOOS=darwin  GOARCH=arm64   go build -o "$OUT_DIR/wch-client-darwin-arm64"     ./cmd/wch-client
GOOS=darwin  GOARCH=amd64   go build -o "$OUT_DIR/wch-client-darwin-amd64"     ./cmd/wch-client
GOOS=linux   GOARCH=amd64   go build -o "$OUT_DIR/wch-client-linux-amd64"      ./cmd/wch-client
GOOS=linux   GOARCH=arm64   go build -o "$OUT_DIR/wch-client-linux-arm64"      ./cmd/wch-client

popd >/dev/null

echo "Done. Binaries at: $OUT_DIR"


