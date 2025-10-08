//go:build sandbox_docker

package sandbox

import (
	"context"
	"fmt"
	"io"
	"os"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

// Runner defines a minimal sandbox interface
type Runner interface {
	Run(ctx context.Context, payload string) (*SandboxResult, error)
}

type noopRunner struct{}

func (n noopRunner) Run(_ context.Context, payload string) (*SandboxResult, error) {
	return &SandboxResult{
		ExitCode:    0,
		Stdout:      payload,
		Duration:    time.Millisecond,
		ThreatScore: 0.0,
		Artifacts:   make(map[string][]byte),
	}, nil
}

type dockerRunner struct{ cli *client.Client }

func (d dockerRunner) Run(ctx context.Context, payload string) (*SandboxResult, error) {
	startTime := time.Now()

	if d.cli == nil {
		return nil, fmt.Errorf("docker client nil")
	}

	img := getenv("SANDBOX_IMAGE", "alpine:latest")
	rc, err := d.cli.ImagePull(ctx, img, dockertypes.ImagePullOptions{})
	if err == nil {
		io.Copy(io.Discard, rc)
		rc.Close()
	}

	cmd := []string{"/bin/sh", "-lc", fmt.Sprintf("printf %q | sha256sum", payload)}
	resp, err := d.cli.ContainerCreate(ctx, &container.Config{
		Image:           img,
		Cmd:             cmd,
		NetworkDisabled: true,
	}, &container.HostConfig{
		AutoRemove: true,
		Resources: container.Resources{
			Memory:   64 * 1024 * 1024, // 64MB limit
			CPUQuota: 50000,            // 50% CPU
		},
	}, nil, nil, "")

	if err != nil {
		return nil, err
	}

	if err := d.cli.ContainerStart(ctx, resp.ID, dockertypes.ContainerStartOptions{}); err != nil {
		return nil, err
	}

	r, err := d.cli.ContainerLogs(ctx, resp.ID, dockertypes.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     true,
	})
	if err != nil {
		return nil, err
	}
	defer r.Close()

	b, _ := io.ReadAll(r)

	return &SandboxResult{
		ExitCode:    0,
		Stdout:      string(b),
		Duration:    time.Since(startTime),
		ThreatScore: calculateThreatScore(payload),
		Artifacts:   make(map[string][]byte),
	}, nil
}

func calculateThreatScore(payload string) float64 {
	score := 0.0

	// Simple heuristics
	if len(payload) > 1000 {
		score += 0.2
	}

	// Check for suspicious patterns
	suspiciousPatterns := []string{"/bin/sh", "exec", "system", "eval"}
	for _, pattern := range suspiciousPatterns {
		if contains(payload, pattern) {
			score += 0.3
		}
	}

	return minFloat(score, 1.0)
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func getenv(k, def string) string {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	return v
}

// NewFromEnv chooses a Runner based on env variables
func NewFromEnv() Runner {
	if os.Getenv("SANDBOX_DOCKER") == "1" {
		cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
		if err == nil {
			return dockerRunner{cli: cli}
		}
	}
	return noopRunner{}
}
