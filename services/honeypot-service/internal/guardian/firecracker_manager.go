package guardian

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// FirecrackerManager provides a minimal lifecycle wrapper around Firecracker microVMs.
type FirecrackerManager struct{}

// StartVM boots a microVM using a kernel and rootfs. This is a thin wrapper for PoC.
func (m *FirecrackerManager) StartVM(ctx context.Context, kernelPath, rootfsPath string, tapIf string, vcpu, memMB int) error {
	// For production: use firecracker-go-sdk. Here we call the binary if available.
	if kernelPath == "" || rootfsPath == "" {
		return fmt.Errorf("kernel and rootfs required")
	}
	cmd := exec.CommandContext(ctx, "firecracker", "--no-api", "--kernel", kernelPath, "--root-drive", rootfsPath)
	return cmd.Start()
}

// StopVM sends a SIGTERM to firecracker; real implementation would manage sockets.
func (m *FirecrackerManager) StopVM(ctx context.Context, pid int) error {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	if err := proc.Signal(syscall.SIGTERM); err != nil {
		return err
	}
	t := time.After(3 * time.Second)
	<-t
	return nil
}
