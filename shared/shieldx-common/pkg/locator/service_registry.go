package locator

// Moved from services/locator to pkg/locator to resolve mixed package issues.
// Original implementation preserved.

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

// ServiceRegistry implements high-performance service discovery
// with health tracking, load balancing, and circuit breaker
type ServiceRegistry struct {
	mu       sync.RWMutex
	services map[string][]*ServiceInstance
	health   map[string]*HealthStatus
	metrics  *RegistryMetrics
}

// ServiceInstance represents a service endpoint
type ServiceInstance struct {
	ID              string            `json:"id"`
	Name            string            `json:"name"`
	Address         string            `json:"address"`
	Port            int               `json:"port"`
	Protocol        string            `json:"protocol"`
	Version         string            `json:"version"`
	Metadata        map[string]string `json:"metadata"`
	RegisteredAt    time.Time         `json:"registered_at"`
	LastHeartbeat   time.Time         `json:"last_heartbeat"`
	Weight          int               `json:"weight"` // For weighted load balancing
	CurrentLoad     int               `json:"current_load"`
	HealthStatus    string            `json:"health_status"`
	CircuitState    string            `json:"circuit_state"`
	FailureCount    int               `json:"failure_count"`
	SuccessCount    int               `json:"success_count"`
	ResponseTimeAvg time.Duration     `json:"response_time_avg"`
}

// HealthStatus tracks service health
type HealthStatus struct {
	Status          string        `json:"status"`
	LastCheck       time.Time     `json:"last_check"`
	ConsecutiveFail int           `json:"consecutive_fail"`
	ResponseTime    time.Duration `json:"response_time"`
	ErrorRate       float64       `json:"error_rate"`
}

// RegistryMetrics tracks registry statistics
type RegistryMetrics struct {
	mu                 sync.RWMutex
	TotalLookups       int64
	CacheHits          int64
	TotalRegistrations int64
	ActiveServices     int
}

// LoadBalancingStrategy defines load balancing algorithm
type LoadBalancingStrategy int

const (
	RoundRobin LoadBalancingStrategy = iota
	LeastConnections
	WeightedRandom
	ResponseTime
	IPHash
)

// NewServiceRegistry creates optimized service registry
func NewServiceRegistry() *ServiceRegistry {
	return &ServiceRegistry{
		services: make(map[string][]*ServiceInstance),
		health:   make(map[string]*HealthStatus),
		metrics:  &RegistryMetrics{},
	}
}

// Register adds service instance to registry
func (sr *ServiceRegistry) Register(instance *ServiceInstance) error {
	if instance.Name == "" || instance.Address == "" {
		return fmt.Errorf("service name and address required")
	}

	sr.mu.Lock()
	defer sr.mu.Unlock()

	// Generate unique ID if not provided
	if instance.ID == "" {
		instance.ID = generateInstanceID(instance.Name, instance.Address, instance.Port)
	}

	instance.RegisteredAt = time.Now()
	instance.LastHeartbeat = time.Now()
	instance.HealthStatus = "healthy"
	instance.CircuitState = "closed"

	if instance.Weight == 0 {
		instance.Weight = 100 // Default weight
	}

	// Add to registry
	sr.services[instance.Name] = append(sr.services[instance.Name], instance)

	// Initialize health status
	sr.health[instance.ID] = &HealthStatus{
		Status:    "healthy",
		LastCheck: time.Now(),
	}

	sr.metrics.TotalRegistrations++
	sr.metrics.ActiveServices = len(sr.services)

	log.Printf("[locator] Registered %s instance: %s (%s:%d)",
		instance.Name, instance.ID, instance.Address, instance.Port)

	return nil
}

// Deregister removes service instance from registry
func (sr *ServiceRegistry) Deregister(instanceID string) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	for serviceName, instances := range sr.services {
		for i, inst := range instances {
			if inst.ID == instanceID {
				// Remove instance
				sr.services[serviceName] = append(instances[:i], instances[i+1:]...)
				delete(sr.health, instanceID)

				if len(sr.services[serviceName]) == 0 {
					delete(sr.services, serviceName)
				}

				log.Printf("[locator] Deregistered instance: %s", instanceID)
				return nil
			}
		}
	}

	return fmt.Errorf("instance not found: %s", instanceID)
}

// Discover finds service instances with load balancing
func (sr *ServiceRegistry) Discover(serviceName string, strategy LoadBalancingStrategy) (*ServiceInstance, error) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	sr.metrics.mu.Lock()
	sr.metrics.TotalLookups++
	sr.metrics.mu.Unlock()

	instances, exists := sr.services[serviceName]
	if !exists || len(instances) == 0 {
		return nil, fmt.Errorf("service not found: %s", serviceName)
	}

	// Filter healthy instances
	healthyInstances := []*ServiceInstance{}
	for _, inst := range instances {
		if sr.isInstanceHealthy(inst) {
			healthyInstances = append(healthyInstances, inst)
		}
	}

	if len(healthyInstances) == 0 {
		return nil, fmt.Errorf("no healthy instances available for: %s", serviceName)
	}

	// Apply load balancing strategy
	var selected *ServiceInstance

	switch strategy {
	case RoundRobin:
		selected = sr.selectRoundRobin(healthyInstances)
	case LeastConnections:
		selected = sr.selectLeastConnections(healthyInstances)
	case WeightedRandom:
		selected = sr.selectWeightedRandom(healthyInstances)
	case ResponseTime:
		selected = sr.selectFastestResponse(healthyInstances)
	default:
		selected = healthyInstances[0]
	}

	// Increment load counter
	selected.CurrentLoad++

	return selected, nil
}

// isInstanceHealthy checks if instance is healthy
func (sr *ServiceRegistry) isInstanceHealthy(inst *ServiceInstance) bool {
	// Check heartbeat timeout
	if time.Since(inst.LastHeartbeat) > 30*time.Second {
		return false
	}

	// Check circuit breaker state
	if inst.CircuitState == "open" {
		return false
	}

	// Check health status
	health, exists := sr.health[inst.ID]
	if !exists {
		return true // Assume healthy if no health data
	}

	return health.Status == "healthy"
}

// selectRoundRobin implements round-robin selection
func (sr *ServiceRegistry) selectRoundRobin(instances []*ServiceInstance) *ServiceInstance {
	// Simple round-robin: select instance with lowest current load
	minLoad := instances[0].CurrentLoad
	selected := instances[0]

	for _, inst := range instances {
		if inst.CurrentLoad < minLoad {
			minLoad = inst.CurrentLoad
			selected = inst
		}
	}

	return selected
}

// selectLeastConnections selects instance with least active connections
func (sr *ServiceRegistry) selectLeastConnections(instances []*ServiceInstance) *ServiceInstance {
	minLoad := instances[0].CurrentLoad
	selected := instances[0]

	for _, inst := range instances {
		if inst.CurrentLoad < minLoad {
			minLoad = inst.CurrentLoad
			selected = inst
		}
	}

	return selected
}

// selectWeightedRandom implements weighted random selection
func (sr *ServiceRegistry) selectWeightedRandom(instances []*ServiceInstance) *ServiceInstance {
	totalWeight := 0
	for _, inst := range instances {
		totalWeight += inst.Weight
	}

	if totalWeight == 0 {
		return instances[0]
	}

	// Generate random number
	random := int(time.Now().UnixNano() % int64(totalWeight))
	cumulative := 0

	for _, inst := range instances {
		cumulative += inst.Weight
		if random < cumulative {
			return inst
		}
	}

	return instances[0]
}

// selectFastestResponse selects instance with best response time
func (sr *ServiceRegistry) selectFastestResponse(instances []*ServiceInstance) *ServiceInstance {
	fastest := instances[0]
	minResponseTime := fastest.ResponseTimeAvg

	for _, inst := range instances {
		if inst.ResponseTimeAvg > 0 && inst.ResponseTimeAvg < minResponseTime {
			minResponseTime = inst.ResponseTimeAvg
			fastest = inst
		}
	}

	return fastest
}

// Heartbeat updates instance last heartbeat time
func (sr *ServiceRegistry) Heartbeat(instanceID string) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	for _, instances := range sr.services {
		for _, inst := range instances {
			if inst.ID == instanceID {
				inst.LastHeartbeat = time.Now()
				return nil
			}
		}
	}

	return fmt.Errorf("instance not found: %s", instanceID)
}

// UpdateHealth updates service health status
func (sr *ServiceRegistry) UpdateHealth(instanceID string, healthy bool, responseTime time.Duration) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	health, exists := sr.health[instanceID]
	if !exists {
		health = &HealthStatus{}
		sr.health[instanceID] = health
	}

	health.LastCheck = time.Now()
	health.ResponseTime = responseTime

	if healthy {
		health.Status = "healthy"
		health.ConsecutiveFail = 0
	} else {
		health.ConsecutiveFail++
		if health.ConsecutiveFail >= 3 {
			health.Status = "unhealthy"
			sr.openCircuitBreaker(instanceID)
		}
	}
}

// openCircuitBreaker opens circuit for unhealthy instance
func (sr *ServiceRegistry) openCircuitBreaker(instanceID string) {
	for _, instances := range sr.services {
		for _, inst := range instances {
			if inst.ID == instanceID {
				inst.CircuitState = "open"
				inst.FailureCount++

				// Schedule circuit breaker reset
				go func() {
					time.Sleep(30 * time.Second)
					sr.resetCircuitBreaker(instanceID)
				}()

				log.Printf("[locator] Circuit breaker opened for instance: %s", instanceID)
				return
			}
		}
	}
}

// resetCircuitBreaker resets circuit to half-open state
func (sr *ServiceRegistry) resetCircuitBreaker(instanceID string) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	for _, instances := range sr.services {
		for _, inst := range instances {
			if inst.ID == instanceID && inst.CircuitState == "open" {
				inst.CircuitState = "half-open"
				log.Printf("[locator] Circuit breaker reset to half-open: %s", instanceID)
				return
			}
		}
	}
}

// ListServices returns all registered services
func (sr *ServiceRegistry) ListServices() map[string][]ServiceInstance {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	result := make(map[string][]ServiceInstance)

	for name, instances := range sr.services {
		serviceInstances := make([]ServiceInstance, len(instances))
		for i, inst := range instances {
			serviceInstances[i] = *inst
		}
		result[name] = serviceInstances
	}

	return result
}

// GetMetrics returns registry metrics
func (sr *ServiceRegistry) GetMetrics() map[string]interface{} {
	sr.metrics.mu.RLock()
	defer sr.metrics.mu.RUnlock()

	cacheHitRate := 0.0
	if sr.metrics.TotalLookups > 0 {
		cacheHitRate = float64(sr.metrics.CacheHits) / float64(sr.metrics.TotalLookups) * 100
	}

	return map[string]interface{}{
		"total_lookups":       sr.metrics.TotalLookups,
		"cache_hits":          sr.metrics.CacheHits,
		"cache_hit_rate":      cacheHitRate,
		"total_registrations": sr.metrics.TotalRegistrations,
		"active_services":     sr.metrics.ActiveServices,
	}
}

// CleanupStaleInstances removes instances with expired heartbeats
func (sr *ServiceRegistry) CleanupStaleInstances() {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	staleTimeout := 60 * time.Second
	now := time.Now()
	removedCount := 0

	for serviceName, instances := range sr.services {
		activeInstances := []*ServiceInstance{}

		for _, inst := range instances {
			if now.Sub(inst.LastHeartbeat) < staleTimeout {
				activeInstances = append(activeInstances, inst)
			} else {
				delete(sr.health, inst.ID)
				removedCount++
				log.Printf("[locator] Removed stale instance: %s (service: %s)", inst.ID, serviceName)
			}
		}

		if len(activeInstances) > 0 {
			sr.services[serviceName] = activeInstances
		} else {
			delete(sr.services, serviceName)
		}
	}

	if removedCount > 0 {
		log.Printf("[locator] Cleaned up %d stale instances", removedCount)
		sr.metrics.ActiveServices = len(sr.services)
	}
}

// StartCleanupRoutine starts background cleanup task
func (sr *ServiceRegistry) StartCleanupRoutine(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			sr.CleanupStaleInstances()
		case <-ctx.Done():
			return
		}
	}
}

// generateInstanceID creates unique instance ID
func generateInstanceID(name, address string, port int) string {
	return fmt.Sprintf("%s-%s-%d-%d", name, address, port, time.Now().UnixNano()%10000)
}

// ExportToJSON exports registry state to JSON
func (sr *ServiceRegistry) ExportToJSON() (string, error) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	data := map[string]interface{}{
		"services":    sr.services,
		"health":      sr.health,
		"metrics":     sr.GetMetrics(),
		"exported_at": time.Now(),
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to export to JSON: %w", err)
	}

	return string(jsonData), nil
}
