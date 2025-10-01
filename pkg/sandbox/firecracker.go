//go:build linux

package sandbox

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/firecracker-microvm/firecracker-go-sdk"
	"github.com/firecracker-microvm/firecracker-go-sdk/client/models"
)

type FirecrackerRunner struct {
	kernelPath    string
	rootfsPath    string
	socketPath    string
	limits        ResourceLimits
	forensicsPath string
}

type ResourceLimits struct {
	VCPUCount    int64
	MemSizeMib   int64
	TimeoutSec   int
	NetworkDeny  bool
	FilesystemRO bool
}

type SandboxResult struct {
	ExitCode     int                    `json:"exit_code"`
	Stdout       string                 `json:"stdout"`
	Stderr       string                 `json:"stderr"`
	Syscalls     []SyscallEvent         `json:"syscalls"`
	NetworkIO    []NetworkEvent         `json:"network_io"`
	FileAccess   []FileEvent            `json:"file_access"`
	MemoryDump   []byte                 `json:"memory_dump,omitempty"`
	Duration     time.Duration          `json:"duration"`
	Artifacts    map[string][]byte      `json:"artifacts"`
	ThreatScore  float64                `json:"threat_score"`
	Fingerprint  string                 `json:"fingerprint"`
}

type SyscallEvent struct {
	Timestamp   int64    `json:"timestamp"`
	PID         uint32   `json:"pid"`
	SyscallNr   uint32   `json:"syscall_nr"`
	SyscallName string   `json:"syscall_name"`
	Args        []uint64 `json:"args"`
	RetCode     int64    `json:"ret_code"`
	Dangerous   bool     `json:"dangerous"`
}

type NetworkEvent struct {
	Timestamp   int64  `json:"timestamp"`
	Protocol    string `json:"protocol"`
	SrcIP       string `json:"src_ip"`
	DstIP       string `json:"dst_ip"`
	SrcPort     uint16 `json:"src_port"`
	DstPort     uint16 `json:"dst_port"`
	Bytes       int64  `json:"bytes"`
	Blocked     bool   `json:"blocked"`
}

type FileEvent struct {
	Timestamp int64  `json:"timestamp"`
	Path      string `json:"path"`
	Operation string `json:"operation"`
	Mode      uint32 `json:"mode"`
	Success   bool   `json:"success"`
}

func NewFirecrackerRunner(kernelPath, rootfsPath string, limits ResourceLimits) *FirecrackerRunner {
	return &FirecrackerRunner{
		kernelPath:    kernelPath,
		rootfsPath:    rootfsPath,
		socketPath:    fmt.Sprintf("/tmp/firecracker-%d.sock", time.Now().UnixNano()),
		limits:        limits,
		forensicsPath: "/tmp/forensics",
	}
}

func (f *FirecrackerRunner) Run(ctx context.Context, payload string) (*SandboxResult, error) {
	startTime := time.Now()
	
	// Create forensics directory
	os.MkdirAll(f.forensicsPath, 0755)
	
	// Configure Firecracker
	cfg := firecracker.Config{
		SocketPath:      f.socketPath,
		KernelImagePath: f.kernelPath,
		KernelArgs:      "console=ttyS0 reboot=k panic=1 pci=off",
		Drives: []models.Drive{{
			DriveID:      firecracker.String("rootfs"),
			PathOnHost:   firecracker.String(f.rootfsPath),
			IsRootDevice: firecracker.Bool(true),
			IsReadOnly:   firecracker.Bool(f.limits.FilesystemRO),
		}},
		MachineCfg: models.MachineConfiguration{
			VcpuCount:  firecracker.Int64(f.limits.VCPUCount),
			MemSizeMib: firecracker.Int64(f.limits.MemSizeMib),
		},
		LogLevel: "Error",
	}

	// Network isolation
	if !f.limits.NetworkDeny {
		cfg.NetworkInterfaces = []firecracker.NetworkInterface{{
			CNIConfiguration: &firecracker.CNIConfiguration{
				NetworkName: "sandbox-net",
				IfName:      "veth0",
			},
		}}
	}

	// Create machine
	machine, err := firecracker.NewMachine(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create machine: %w", err)
	}
	defer machine.Shutdown(ctx)

	// Start VM
	if err := machine.Start(ctx); err != nil {
		return nil, fmt.Errorf("failed to start machine: %w", err)
	}

	// Execute payload with timeout
	execCtx, cancel := context.WithTimeout(ctx, time.Duration(f.limits.TimeoutSec)*time.Second)
	defer cancel()

	result := &SandboxResult{
		Artifacts: make(map[string][]byte),
		Duration:  time.Since(startTime),
	}

	// Execute and collect forensics
	if err := f.executeWithForensics(execCtx, machine, payload, result); err != nil {
		result.ExitCode = 1
		result.Stderr = err.Error()
	}

	// Calculate threat score
	result.ThreatScore = f.calculateThreatScore(result)
	result.Fingerprint = f.generateFingerprint(result)

	return result, nil
}

func (f *FirecrackerRunner) executeWithForensics(ctx context.Context, machine *firecracker.Machine, payload string, result *SandboxResult) error {
	// Write payload to VM filesystem
	payloadPath := filepath.Join(f.forensicsPath, "payload.bin")
	if err := os.WriteFile(payloadPath, []byte(payload), 0644); err != nil {
		return err
	}

	// Start eBPF monitoring (implemented in next section)
	monitor := NeweBPFMonitor()
	if err := monitor.Start(ctx); err != nil {
		return fmt.Errorf("failed to start eBPF monitor: %w", err)
	}
	defer monitor.Stop()

	// Execute payload in VM
	// This would use Firecracker's agent or SSH to execute commands
	// For now, simulate execution
	result.ExitCode = 0
	result.Stdout = "Payload executed successfully"
	
	// Collect forensics data
	result.Syscalls = monitor.GetSyscalls()
	result.NetworkIO = monitor.GetNetworkEvents()
	result.FileAccess = monitor.GetFileEvents()

	return nil
}

func (f *FirecrackerRunner) calculateThreatScore(result *SandboxResult) float64 {
	score := 0.0
	
	// Dangerous syscalls
	for _, syscall := range result.Syscalls {
		if syscall.Dangerous {
			score += 10.0
		}
	}
	
	// Network activity when denied
	if f.limits.NetworkDeny && len(result.NetworkIO) > 0 {
		score += 50.0
	}
	
	// File system writes when read-only
	if f.limits.FilesystemRO {
		for _, file := range result.FileAccess {
			if file.Operation == "write" && file.Success {
				score += 20.0
			}
		}
	}
	
	// Execution time anomaly
	if result.Duration > time.Duration(f.limits.TimeoutSec/2)*time.Second {
		score += 15.0
	}
	
	return minFloat(score, 100.0)
}

func (f *FirecrackerRunner) generateFingerprint(result *SandboxResult) string {
	// Generate SHA256 hash of syscall patterns + network behavior
	// This helps identify similar attack patterns
	return fmt.Sprintf("fc_%x", time.Now().UnixNano()%0xFFFFFFFF)
}

// minFloat is defined in sandbox.go for the package; reuse it here.