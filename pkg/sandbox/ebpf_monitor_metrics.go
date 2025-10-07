//go:build linux

package sandbox

import (
	"time"
	"shieldx/shared/metrics"
)

// MonitorWithMetrics wraps eBPFMonitor with metrics instrumentation
type MonitorWithMetrics struct {
	*eBPFMonitor
	
	// Metrics with service and sandbox labels
	syscallCounter    *metrics.LabeledCounter
	syscallLatency    *metrics.LabeledHistogram
	networkBytesIn    *metrics.LabeledCounter
	networkBytesOut   *metrics.LabeledCounter
	fileOpsCounter    *metrics.LabeledCounter
	dangerousSyscalls *metrics.LabeledCounter
}

// NewMonitorWithMetrics creates a new eBPF monitor with metrics instrumentation
func NewMonitorWithMetrics(registry *metrics.Registry) *MonitorWithMetrics {
	baseMonitor := NeweBPFMonitor()
	
	// Counter for syscall events by service/sandbox/type
	syscallCounter := metrics.NewLabeledCounter(
		"ebpf_syscall_total",
		"Total syscalls monitored by eBPF",
		[]string{"service", "sandbox", "syscall"},
	)
	
	// Histogram for syscall latency
	syscallLatency := metrics.NewLabeledHistogram(
		"ebpf_syscall_duration_seconds",
		"Syscall execution duration",
		[]string{"service", "sandbox", "syscall"},
		[]float64{0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
	)
	
	// Network bytes counters
	networkBytesIn := metrics.NewLabeledCounter(
		"ebpf_network_bytes_received_total",
		"Total bytes received by sandboxed process",
		[]string{"service", "sandbox", "protocol"},
	)
	
	networkBytesOut := metrics.NewLabeledCounter(
		"ebpf_network_bytes_sent_total",
		"Total bytes sent by sandboxed process",
		[]string{"service", "sandbox", "protocol"},
	)
	
	// File operations counter
	fileOpsCounter := metrics.NewLabeledCounter(
		"ebpf_file_operations_total",
		"Total file operations by sandboxed process",
		[]string{"service", "sandbox", "operation"},
	)
	
	// Dangerous syscalls counter (security monitoring)
	dangerousSyscalls := metrics.NewLabeledCounter(
		"ebpf_dangerous_syscalls_total",
		"Count of potentially dangerous syscalls",
		[]string{"service", "sandbox", "syscall"},
	)
	
	// Register with metrics registry if provided
	if registry != nil {
		registry.RegisterLabeledCounter(syscallCounter)
		registry.RegisterLabeledHistogram(syscallLatency)
		registry.RegisterLabeledCounter(networkBytesIn)
		registry.RegisterLabeledCounter(networkBytesOut)
		registry.RegisterLabeledCounter(fileOpsCounter)
		registry.RegisterLabeledCounter(dangerousSyscalls)
	}
	
	return &MonitorWithMetrics{
		eBPFMonitor:       baseMonitor,
		syscallCounter:    syscallCounter,
		syscallLatency:    syscallLatency,
		networkBytesIn:    networkBytesIn,
		networkBytesOut:   networkBytesOut,
		fileOpsCounter:    fileOpsCounter,
		dangerousSyscalls: dangerousSyscalls,
	}
}

// RecordSyscallWithMetrics records a syscall event with metrics
func (m *MonitorWithMetrics) RecordSyscallWithMetrics(
	serviceID, sandboxID, syscall string, 
	duration time.Duration, 
	isDangerous bool,
) {
	// Update metrics with labels
	labels := map[string]string{"service": serviceID, "sandbox": sandboxID, "syscall": syscall}
	m.syscallCounter.Inc(labels)
	m.syscallLatency.Observe(labels, duration.Seconds())
	
	// Track dangerous syscalls separately for security monitoring
	if isDangerous {
	m.dangerousSyscalls.Inc(labels)
	}
	
	// Track specific syscall types for file operations
	switch syscall {
	case "read", "write", "open", "close", "stat", "fstat", "lstat", 
	     "openat", "readv", "writev", "pread64", "pwrite64":
	m.fileOpsCounter.Inc(labels)
	}
}

// RecordNetworkActivityWithMetrics records network bytes transferred
func (m *MonitorWithMetrics) RecordNetworkActivityWithMetrics(
	serviceID, sandboxID, protocol string, 
	bytesIn, bytesOut uint64,
) {
	if bytesIn > 0 {
		// Aggregate byte counters using Add only (no extra Inc to avoid double counting)
		m.networkBytesIn.Add(map[string]string{"service": serviceID, "sandbox": sandboxID, "protocol": protocol}, bytesIn)
	}
	if bytesOut > 0 {
		m.networkBytesOut.Add(map[string]string{"service": serviceID, "sandbox": sandboxID, "protocol": protocol}, bytesOut)
	}
}

// GetMetricsSummary returns a summary of metrics for a specific service/sandbox
func (m *MonitorWithMetrics) GetMetricsSummary(serviceID, sandboxID string) map[string]interface{} {
	return map[string]interface{}{
		"service":    serviceID,
		"sandbox":    sandboxID,
		"timestamp":  time.Now().Unix(),
		"metrics": map[string]string{
			"syscalls":   "ebpf_syscall_total{service=\"" + serviceID + "\",sandbox=\"" + sandboxID + "\"}",
			"latency":    "ebpf_syscall_duration_seconds{service=\"" + serviceID + "\",sandbox=\"" + sandboxID + "\"}",
			"network_in": "ebpf_network_bytes_received_total{service=\"" + serviceID + "\",sandbox=\"" + sandboxID + "\"}",
			"network_out": "ebpf_network_bytes_sent_total{service=\"" + serviceID + "\",sandbox=\"" + sandboxID + "\"}",
			"file_ops":   "ebpf_file_operations_total{service=\"" + serviceID + "\",sandbox=\"" + sandboxID + "\"}",
			"dangerous":  "ebpf_dangerous_syscalls_total{service=\"" + serviceID + "\",sandbox=\"" + sandboxID + "\"}",
		},
	}
}
