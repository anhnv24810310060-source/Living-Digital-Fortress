FROM golang:1.22-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata gcc musl-dev

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application with optimizations
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo \
    -ldflags '-extldflags "-static" -s -w' \
    -o pqc-service .

FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates curl

WORKDIR /root/

# Copy the binary
COPY --from=builder /app/pqc-service .

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Create data directory
RUN mkdir -p /app/data /app/config

# Create non-root user
RUN addgroup -g 1001 pqc && \
    adduser -D -s /bin/sh -u 1001 -G pqc pqc

# Change ownership
RUN chown -R pqc:pqc /app /root

# Switch to non-root user
USER pqc

# Expose port
EXPOSE 8092

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8092/health || exit 1

# Run the application
ENTRYPOINT ["./pqc-service"]