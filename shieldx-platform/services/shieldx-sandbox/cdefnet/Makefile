# CDefNet Makefile

.PHONY: all build test clean run docker-up docker-down

# Build the service
build:
	go build -o bin/cdefnet .

# Run tests
test:
	go test -v ./privacy/...
	go test -v ./store/...

# Run the service locally
run:
	go run .

# Clean build artifacts
clean:
	rm -rf bin/

# Docker operations
docker-build:
	docker build -t shieldx/cdefnet .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f cdefnet

# Database operations
db-migrate:
	docker-compose exec postgres psql -U cdefnet -d cdefnet -f /docker-entrypoint-initdb.d/init.sql

db-shell:
	docker-compose exec postgres psql -U cdefnet -d cdefnet

# API testing
test-api:
	chmod +x test_api.sh
	./test_api.sh

# Development setup
dev-setup:
	go mod tidy
	go mod download

# Production deployment
deploy:
	docker-compose -f docker-compose.prod.yml up -d

# Week 1 acceptance test
week1-test: docker-up
	@echo "Waiting for services to start..."
	sleep 10
	@echo "Running Week 1 acceptance tests..."
	./test_api.sh
	@echo "Week 1 CDefNet skeleton complete!"

all: build test