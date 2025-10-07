FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o shadow-service .

FROM alpine:latest

RUN apk --no-cache add ca-certificates wget
WORKDIR /root/

COPY --from=builder /app/shadow-service .

EXPOSE 5005

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5005/health || exit 1

CMD ["./shadow-service"]
