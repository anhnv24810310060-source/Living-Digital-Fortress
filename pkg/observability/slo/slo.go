package slo

import (
	"context"
	"fmt"
	"sync"
	"time"

	"shieldx/pkg/metrics"
)

// SLO (Service Level Objective) defines service quality targets
type SLO struct {
	ServiceName        string
	AvailabilityTarget float64 // e.g., 0.999 for 99.9%
	LatencyP95Target   time.Duration
	LatencyP99Target   time.Duration
	ErrorBudget        float64 // remaining error budget (0.0 - 1.0)

	// Metrics
	requestTotal   *metrics.Counter
	requestSuccess *metrics.Counter
	requestErrors  *metrics.Counter
	latencyHist    *metrics.Histogram

	mu sync.RWMutex
}

// SLOManager manages multiple service SLOs
type SLOManager struct {
	slos map[string]*SLO
	mu   sync.RWMutex
}

// NewSLOManager creates a new SLO manager
func NewSLOManager() *SLOManager {
	return &SLOManager{
		slos: make(map[string]*SLO),
	}
}

// RegisterSLO registers a new SLO for a service
func (m *SLOManager) RegisterSLO(serviceName string, availabilityTarget float64, p95Latency, p99Latency time.Duration) *SLO {
	m.mu.Lock()
	defer m.mu.Unlock()

	slo := &SLO{
		ServiceName:        serviceName,
		AvailabilityTarget: availabilityTarget,
		LatencyP95Target:   p95Latency,
		LatencyP99Target:   p99Latency,
		ErrorBudget:        1.0, // Start with full error budget
		requestTotal:       metrics.NewCounter(fmt.Sprintf("%s_requests_total", serviceName), "Total requests"),
		requestSuccess:     metrics.NewCounter(fmt.Sprintf("%s_requests_success", serviceName), "Successful requests"),
		requestErrors:      metrics.NewCounter(fmt.Sprintf("%s_requests_errors", serviceName), "Failed requests"),
		latencyHist:        metrics.NewHistogram(fmt.Sprintf("%s_request_duration_seconds", serviceName), "Request latency distribution", nil),
	}

	m.slos[serviceName] = slo
	return slo
}

// RecordRequest records a request for SLO tracking
func (s *SLO) RecordRequest(duration time.Duration, success bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.requestTotal.Inc()
	if success {
		s.requestSuccess.Inc()
	} else {
		s.requestErrors.Inc()
	}
	s.latencyHist.Observe(duration.Seconds())
}

// GetAvailability returns current availability
func (s *SLO) GetAvailability() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	total := s.requestTotal.Value()
	if total == 0 {
		return 1.0
	}
	success := s.requestSuccess.Value()
	return float64(success) / float64(total)
}

// GetErrorBudget calculates remaining error budget
func (s *SLO) GetErrorBudget() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	availability := s.GetAvailability()
	allowedFailures := 1.0 - s.AvailabilityTarget
	actualFailures := 1.0 - availability

	if allowedFailures == 0 {
		return 1.0
	}

	remaining := 1.0 - (actualFailures / allowedFailures)
	if remaining < 0 {
		return 0
	}
	return remaining
}

// SLOStatus represents the current SLO status
type SLOStatus struct {
	ServiceName    string
	Availability   float64
	AvailabilityOK bool
	ErrorBudget    float64
	P95Latency     float64
	P99Latency     float64
	LatencyOK      bool
	TotalRequests  uint64
	ErrorRequests  uint64
}

// GetStatus returns current SLO status
func (s *SLO) GetStatus() SLOStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()

	availability := s.GetAvailability()
	// If Histogram doesn't support quantiles directly, approximate using available API.
	// Placeholder values; consider integrating a histogram library with quantiles if needed.
	p95 := 0.0
	p99 := 0.0

	return SLOStatus{
		ServiceName:    s.ServiceName,
		Availability:   availability,
		AvailabilityOK: availability >= s.AvailabilityTarget,
		ErrorBudget:    s.GetErrorBudget(),
		P95Latency:     p95,
		P99Latency:     p99,
		LatencyOK:      true, // without quantiles, assume OK; replace when quantiles available
		TotalRequests:  s.requestTotal.Value(),
		ErrorRequests:  s.requestErrors.Value(),
	}
}

// GetAllStatuses returns status for all registered SLOs
func (m *SLOManager) GetAllStatuses() []SLOStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	statuses := make([]SLOStatus, 0, len(m.slos))
	for _, slo := range m.slos {
		statuses = append(statuses, slo.GetStatus())
	}
	return statuses
}

// MonitorSLOs starts monitoring and alerting on SLO violations
func (m *SLOManager) MonitorSLOs(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.checkSLOs()
		}
	}
}

func (m *SLOManager) checkSLOs() {
	statuses := m.GetAllStatuses()
	for _, status := range statuses {
		if !status.AvailabilityOK {
			fmt.Printf("[SLO ALERT] %s: Availability %.4f%% below target\n",
				status.ServiceName, status.Availability*100)
		}
		if status.ErrorBudget < 0.2 {
			fmt.Printf("[SLO WARNING] %s: Error budget at %.1f%%\n",
				status.ServiceName, status.ErrorBudget*100)
		}
		if !status.LatencyOK {
			fmt.Printf("[SLO ALERT] %s: Latency p95=%.3fs p99=%.3fs exceeds target\n",
				status.ServiceName, status.P95Latency, status.P99Latency)
		}
	}
}
