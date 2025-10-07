# Makefile for PERSON 3 - Business Logic & Infrastructure Services
# ✅ PHẢI backup database trước migrations
# ✅ PHẢI test rules trong shadow trước deploy

.PHONY: all build test clean run-shadow run-credits deploy help

# Variables
SHADOW_SERVICE := services/shadow
CREDITS_SERVICE := services/credits
DIGITAL_TWIN_SERVICE := services/digital_twin
MARKETPLACE_SERVICE := services/marketplace

SHADOW_BINARY := bin/shadow-service
CREDITS_BINARY := bin/credits-service

GO := go
GOFLAGS := -v
PORT_SHADOW := 7070
PORT_CREDITS := 5002

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

all: build ## Build all services

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

build: build-shadow build-credits ## Build all services
	@echo "$(GREEN)✓ All services built successfully$(NC)"

build-shadow: ## Build shadow service
	@echo "$(YELLOW)Building Shadow Service...$(NC)"
	@cd $(SHADOW_SERVICE) && $(GO) build $(GOFLAGS) -o ../../$(SHADOW_BINARY) .
	@echo "$(GREEN)✓ Shadow Service built: $(SHADOW_BINARY)$(NC)"

build-credits: ## Build credits service
	@echo "$(YELLOW)Building Credits Service...$(NC)"
	@cd $(CREDITS_SERVICE) && $(GO) build $(GOFLAGS) -o ../../$(CREDITS_BINARY) .
	@echo "$(GREEN)✓ Credits Service built: $(CREDITS_BINARY)$(NC)"

##@ Test

test: ## Run all tests
	@echo "$(YELLOW)Running tests for all PERSON 3 services...$(NC)"
	@$(GO) test -v ./$(SHADOW_SERVICE)/... || exit 1
	@$(GO) test -v ./$(CREDITS_SERVICE)/... || exit 1
	@$(GO) test -v ./$(DIGITAL_TWIN_SERVICE)/... || exit 1
	@$(GO) test -v ./$(MARKETPLACE_SERVICE)/... || exit 1
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-shadow: ## Run shadow service tests
	@echo "$(YELLOW)Testing Shadow Service...$(NC)"
	@$(GO) test -v ./$(SHADOW_SERVICE)/...
	@echo "$(GREEN)✓ Shadow tests passed$(NC)"

test-credits: ## Run credits service tests
	@echo "$(YELLOW)Testing Credits Service...$(NC)"
	@$(GO) test -v ./$(CREDITS_SERVICE)/...
	@echo "$(GREEN)✓ Credits tests passed$(NC)"

test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	@$(GO) test -v -tags=integration ./...
	@echo "$(GREEN)✓ Integration tests passed$(NC)"

##@ Run

run-shadow: build-shadow ## Run shadow service
	@echo "$(GREEN)Starting Shadow Service on port $(PORT_SHADOW)...$(NC)"
	@PORT=$(PORT_SHADOW) ./$(SHADOW_BINARY)

run-credits: build-credits ## Run credits service
	@echo "$(GREEN)Starting Credits Service on port $(PORT_CREDITS)...$(NC)"
	@PORT=$(PORT_CREDITS) ./$(CREDITS_BINARY)

run-all: build ## Run all services concurrently
	@echo "$(GREEN)Starting all PERSON 3 services...$(NC)"
	@echo "$(YELLOW)Shadow Service: http://localhost:$(PORT_SHADOW)$(NC)"
	@echo "$(YELLOW)Credits Service: http://localhost:$(PORT_CREDITS)$(NC)"
	@PORT=$(PORT_SHADOW) ./$(SHADOW_BINARY) & \
	PORT=$(PORT_CREDITS) ./$(CREDITS_BINARY) & \
	wait

##@ Docker

docker-build: ## Build Docker images
	@echo "$(YELLOW)Building Docker images...$(NC)"
	@docker build -t shadow-service:latest -f docker/Dockerfile.shadow .
	@docker build -t credits-service:latest -f docker/Dockerfile.credits .
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-run: ## Run services in Docker
	@echo "$(GREEN)Starting services with Docker Compose...$(NC)"
	@docker-compose -f docker-compose.person3.yml up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "Shadow Service: http://localhost:7070"
	@echo "Credits Service: http://localhost:5002"

docker-stop: ## Stop Docker services
	@docker-compose -f docker-compose.person3.yml down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs: ## Show Docker logs
	@docker-compose -f docker-compose.person3.yml logs -f

##@ Database

db-backup: ## Backup database (✅ REQUIRED before migrations)
	@echo "$(YELLOW)Creating database backup...$(NC)"
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	docker exec -t postgres pg_dump -U fortress fortress > backups/fortress_$$timestamp.sql
	@echo "$(GREEN)✓ Database backed up to backups/fortress_$$timestamp.sql$(NC)"

db-restore: ## Restore database from latest backup
	@echo "$(YELLOW)Restoring database from latest backup...$(NC)"
	@latest=$$(ls -t backups/*.sql | head -1); \
	docker exec -i postgres psql -U fortress fortress < $$latest
	@echo "$(GREEN)✓ Database restored from $$latest$(NC)"

db-migrate: db-backup ## Run database migrations (with backup)
	@echo "$(YELLOW)Running database migrations...$(NC)"
	@$(GO) run migrations/migrate.go up
	@echo "$(GREEN)✓ Migrations completed$(NC)"

##@ Deployment

shadow-test: ## Test rules in shadow mode (✅ REQUIRED before deploy)
	@echo "$(YELLOW)Testing rules in shadow environment...$(NC)"
	@curl -X POST http://localhost:5005/shadow/evaluate \
		-H "Content-Type: application/json" \
		-d '{"rule_id": "test_rule", "test_percent": 10}' || \
		(echo "$(RED)✗ Shadow test failed$(NC)" && exit 1)
	@echo "$(GREEN)✓ Shadow test passed$(NC)"

chaos-test: ## Run chaos engineering tests
	@echo "$(YELLOW)Running chaos engineering tests...$(NC)"
	@curl -X POST http://localhost:7070/api/v1/chaos/enable
	@curl -X POST http://localhost:7070/api/v1/chaos/experiments/exp-001/run
	@sleep 35
	@curl http://localhost:7070/api/v1/chaos/metrics
	@echo "$(GREEN)✓ Chaos test completed$(NC)"

canary-deploy: shadow-test ## Deploy with canary strategy
	@echo "$(YELLOW)Deploying with canary strategy...$(NC)"
	@curl -X POST http://localhost:7070/api/v1/deploy \
		-H "Content-Type: application/json" \
		-d '{"service": "my-service", "version": "v2.0.0", "strategy": 1}'
	@echo "$(GREEN)✓ Canary deployment started$(NC)"

##@ Health Checks

health: ## Check health of all services
	@echo "$(YELLOW)Checking service health...$(NC)"
	@echo -n "Shadow Service: "
	@curl -s http://localhost:7070/api/v1/health | grep -q "healthy" && \
		echo "$(GREEN)✓ Healthy$(NC)" || echo "$(RED)✗ Unhealthy$(NC)"
	@echo -n "Credits Service: "
	@curl -s http://localhost:5002/health | grep -q "healthy" && \
		echo "$(GREEN)✓ Healthy$(NC)" || echo "$(RED)✗ Unhealthy$(NC)"

metrics: ## Display metrics for all services
	@echo "$(YELLOW)=== Shadow Service Metrics ===$(NC)"
	@curl -s http://localhost:7070/api/v1/chaos/metrics | jq '.'
	@echo ""
	@echo "$(YELLOW)=== DR Status ===$(NC)"
	@curl -s http://localhost:7070/api/v1/dr/status | jq '.'
	@echo ""
	@echo "$(YELLOW)=== Sharding Metrics ===$(NC)"
	@curl -s http://localhost:7070/api/v1/sharding/metrics | jq '.'

##@ Clean

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@rm -rf bin/
	@rm -f $(SHADOW_SERVICE)/shadow-service
	@rm -f $(CREDITS_SERVICE)/credits-service
	@echo "$(GREEN)✓ Clean complete$(NC)"

clean-all: clean ## Clean all artifacts including Docker volumes
	@echo "$(YELLOW)Cleaning all artifacts...$(NC)"
	@docker-compose -f docker-compose.person3.yml down -v
	@echo "$(GREEN)✓ All artifacts cleaned$(NC)"

##@ Development

dev: ## Run in development mode with hot reload
	@echo "$(GREEN)Starting development mode...$(NC)"
	@which air > /dev/null || go install github.com/cosmtrek/air@latest
	@air -c .air.toml

lint: ## Run linters
	@echo "$(YELLOW)Running linters...$(NC)"
	@which golangci-lint > /dev/null || curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $$(go env GOPATH)/bin
	@golangci-lint run ./...
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	@gofmt -w -s .
	@goimports -w .
	@echo "$(GREEN)✓ Code formatted$(NC)"

##@ Production

production-deploy: db-backup shadow-test chaos-test ## Full production deployment pipeline
	@echo "$(YELLOW)=== Starting Production Deployment Pipeline ===$(NC)"
	@echo "$(GREEN)✓ Database backed up$(NC)"
	@echo "$(GREEN)✓ Shadow testing completed$(NC)"
	@echo "$(GREEN)✓ Chaos testing completed$(NC)"
	@echo "$(YELLOW)Proceeding with canary deployment...$(NC)"
	@make canary-deploy
	@echo "$(GREEN)✓ Production deployment pipeline completed$(NC)"

production-rollback: ## Rollback production deployment
	@echo "$(RED)Rolling back production deployment...$(NC)"
	@curl -X POST http://localhost:7070/api/v1/deploy/$$(cat .last_deploy_id)/rollback \
		-H "Content-Type: application/json" \
		-d '{"reason": "manual rollback"}'
	@echo "$(GREEN)✓ Rollback completed$(NC)"

##@ Monitoring

logs-shadow: ## Tail shadow service logs
	@docker logs -f shadow-service

logs-credits: ## Tail credits service logs
	@docker logs -f credits-service

stats: ## Show system statistics
	@echo "$(YELLOW)=== System Statistics ===$(NC)"
	@echo "Services:"
	@ps aux | grep -E "(shadow-service|credits-service)" | grep -v grep
	@echo ""
	@echo "Memory Usage:"
	@ps aux | grep -E "(shadow-service|credits-service)" | grep -v grep | awk '{sum+=$$4} END {print "Total: " sum "%"}'

##@ Documentation

docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	@which godoc > /dev/null || go install golang.org/x/tools/cmd/godoc@latest
	@echo "$(GREEN)Documentation server starting at http://localhost:6060$(NC)"
	@godoc -http=:6060

api-docs: ## Generate API documentation
	@echo "$(YELLOW)Generating API documentation...$(NC)"
	@which swag > /dev/null || go install github.com/swaggo/swag/cmd/swag@latest
	@swag init -g main.go
	@echo "$(GREEN)✓ API documentation generated$(NC)"

##@ Examples

example-chaos: ## Run chaos engineering example
	@echo "$(YELLOW)=== Chaos Engineering Example ===$(NC)"
	@echo "1. Enable chaos engineering"
	@curl -X POST http://localhost:7070/api/v1/chaos/enable
	@echo ""
	@echo "2. Run service failure experiment"
	@curl -X POST http://localhost:7070/api/v1/chaos/experiments/exp-001/run
	@echo ""
	@echo "3. Check metrics"
	@curl -s http://localhost:7070/api/v1/chaos/metrics | jq '.'

example-dr: ## Run disaster recovery example
	@echo "$(YELLOW)=== Disaster Recovery Example ===$(NC)"
	@echo "1. Check DR status"
	@curl -s http://localhost:7070/api/v1/dr/status | jq '.'
	@echo ""
	@echo "2. Replicate a change"
	@curl -X POST http://localhost:7070/api/v1/dr/replicate \
		-H "Content-Type: application/json" \
		-d '{"type": 1, "entity": "user", "key": "user123", "value": "data"}'
	@echo ""
	@echo "3. Create checkpoint"
	@curl -X POST http://localhost:7070/api/v1/dr/checkpoint/aws-us-east-1

example-deploy: ## Run zero-downtime deployment example
	@echo "$(YELLOW)=== Zero-Downtime Deployment Example ===$(NC)"
	@echo "1. Create deployment"
	@curl -X POST http://localhost:7070/api/v1/deploy \
		-H "Content-Type: application/json" \
		-d '{"service": "example-service", "version": "v2.0.0", "strategy": 1}'
	@echo ""
	@echo "2. Start deployment (will auto-progress through canary stages)"
	@sleep 2

example-sharding: ## Run database sharding example
	@echo "$(YELLOW)=== Database Sharding Example ===$(NC)"
	@echo "1. Check sharding metrics"
	@curl -s http://localhost:7070/api/v1/sharding/metrics | jq '.'
	@echo ""
	@echo "2. Execute query"
	@curl -X POST http://localhost:7070/api/v1/sharding/query \
		-H "Content-Type: application/json" \
		-d '{"type": 0, "table": "users", "shard_key": "user123"}'

.DEFAULT_GOAL := help
