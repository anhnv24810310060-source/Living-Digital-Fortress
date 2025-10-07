//go:build linux

package sandbox

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/perf"
	"github.com/cilium/ebpf/rlimit"
)

type eBPFMonitor struct {
	collection *ebpf.Collection
	links      []link.Link
	perfReader *perf.Reader

	syscalls   []SyscallEvent
	networkIO  []NetworkEvent
	fileAccess []FileEvent

	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
}

// Dangerous syscalls that indicate potential exploit behavior
var dangerousSyscalls = map[uint32]string{
	syscall.SYS_EXECVE:   "execve",
	syscall.SYS_PTRACE:   "ptrace",
	syscall.SYS_PRCTL:    "prctl",
	syscall.SYS_SETUID:   "setuid",
	syscall.SYS_SETGID:   "setgid",
	syscall.SYS_MPROTECT: "mprotect",
	syscall.SYS_MMAP:     "mmap",
	syscall.SYS_SOCKET:   "socket",
	syscall.SYS_CONNECT:  "connect",
	syscall.SYS_BIND:     "bind",
	syscall.SYS_LISTEN:   "listen",
	syscall.SYS_ACCEPT:   "accept",
	syscall.SYS_SENDTO:   "sendto",
	syscall.SYS_RECVFROM: "recvfrom",
	syscall.SYS_CLONE:    "clone",
	syscall.SYS_FORK:     "fork",
	syscall.SYS_VFORK:    "vfork",
	syscall.SYS_KILL:     "kill",
	syscall.SYS_TKILL:    "tkill",
	syscall.SYS_TGKILL:   "tgkill",
}

func NeweBPFMonitor() *eBPFMonitor {
	return &eBPFMonitor{
		syscalls:   make([]SyscallEvent, 0),
		networkIO:  make([]NetworkEvent, 0),
		fileAccess: make([]FileEvent, 0),
	}
}

func (e *eBPFMonitor) Start(ctx context.Context) error {
	e.ctx, e.cancel = context.WithCancel(ctx)

	// Remove memory limit for eBPF
	if err := rlimit.RemoveMemlock(); err != nil {
		return fmt.Errorf("failed to remove memlock: %w", err)
	}

	// Resolve BPF object path (ENV override -> local paths)
	objPath := os.Getenv("EBPF_OBJECT_PATH")
	if objPath == "" {
		// try common locations relative to repo and cwd
		tries := []string{
			"bpf/syscall_tracer.o",
			"pkg/sandbox/bpf/syscall_tracer.o",
			filepath.Join(filepath.Dir(os.Args[0]), "bpf", "syscall_tracer.o"),
		}
		for _, p := range tries {
			if _, err := os.Stat(p); err == nil {
				objPath = p
				break
			}
		}
	}
	if objPath == "" {
		return fmt.Errorf("eBPF object not found; set EBPF_OBJECT_PATH or build bpf/syscall_tracer.o")
	}
	// Load eBPF program
	spec, err := ebpf.LoadCollectionSpec(objPath)
	if err != nil {
		return fmt.Errorf("failed to load eBPF spec from %s: %w", objPath, err)
	}

	e.collection, err = ebpf.NewCollection(spec)
	if err != nil {
		return fmt.Errorf("failed to create eBPF collection: %w", err)
	}

	// Initialize default dangerous syscall set (best-effort)
	_ = e.initDangerousSyscalls()
	// Optionally monitor current process if EBPF_MONITOR_SELF=true (useful for smoke tests)
	if os.Getenv("EBPF_MONITOR_SELF") == "true" {
		_ = e.AddMonitoredPID(uint32(os.Getpid()))
	}

	// Attach tracepoints
	if err := e.attachTracepoints(); err != nil {
		return fmt.Errorf("failed to attach tracepoints: %w", err)
	}

	// Start perf event reader
	if err := e.startPerfReader(); err != nil {
		return fmt.Errorf("failed to start perf reader: %w", err)
	}

	return nil
}

func (e *eBPFMonitor) Stop() {
	if e.cancel != nil {
		e.cancel()
	}

	if e.perfReader != nil {
		e.perfReader.Close()
	}

	for _, l := range e.links {
		l.Close()
	}

	if e.collection != nil {
		e.collection.Close()
	}
}

func (e *eBPFMonitor) attachTracepoints() error {
	// Attach syscall entry tracepoint
	progEnter := e.collection.Programs["trace_syscall_enter"]
	l, err := link.Tracepoint("syscalls", "sys_enter_openat", progEnter, nil)
	if err != nil {
		return err
	}
	e.links = append(e.links, l)

	// Attach syscall exit tracepoint
	progExit := e.collection.Programs["trace_syscall_exit"]
	l, err = link.Tracepoint("syscalls", "sys_exit_openat", progExit, nil)
	if err != nil {
		return err
	}
	e.links = append(e.links, l)

	// Attach network tracepoints
	progNet := e.collection.Programs["trace_network"]
	l, err = link.Tracepoint("syscalls", "sys_enter_socket", progNet, nil)
	if err != nil {
		return err
	}
	e.links = append(e.links, l)

	return nil
}

func (e *eBPFMonitor) startPerfReader() error {
	var err error
	e.perfReader, err = perf.NewReader(e.collection.Maps["events"], os.Getpagesize())
	if err != nil {
		return err
	}

	go e.readEvents()
	return nil
}

func (e *eBPFMonitor) readEvents() {
	for {
		select {
		case <-e.ctx.Done():
			return
		default:
			record, err := e.perfReader.Read()
			if err != nil {
				continue
			}

			e.processEvent(record.RawSample)
		}
	}
}

func (e *eBPFMonitor) processEvent(data []byte) {
	if len(data) < 32 {
		return
	}

	// Parse event structure from eBPF
	eventType := binary.LittleEndian.Uint32(data[0:4])
	timestamp := int64(binary.LittleEndian.Uint64(data[4:12]))
	pid := binary.LittleEndian.Uint32(data[12:16])

	e.mu.Lock()
	defer e.mu.Unlock()

	switch eventType {
	case 1: // Syscall event
		syscallNr := binary.LittleEndian.Uint32(data[16:20])
		retCode := int64(binary.LittleEndian.Uint64(data[20:28]))

		syscallName, dangerous := dangerousSyscalls[syscallNr]
		if syscallName == "" {
			syscallName = fmt.Sprintf("syscall_%d", syscallNr)
		}

		event := SyscallEvent{
			Timestamp:   timestamp,
			PID:         pid,
			SyscallNr:   syscallNr,
			SyscallName: syscallName,
			RetCode:     retCode,
			Dangerous:   dangerous,
		}

		// Parse args if available
		if len(data) >= 80 {
			for i := 0; i < 6; i++ {
				arg := binary.LittleEndian.Uint64(data[28+i*8 : 36+i*8])
				event.Args = append(event.Args, arg)
			}
		}

		e.syscalls = append(e.syscalls, event)

	case 2: // Network event
		protocol := "tcp"
		if len(data) > 20 {
			if data[20] == 17 {
				protocol = "udp"
			}
		}

		event := NetworkEvent{
			Timestamp: timestamp,
			Protocol:  protocol,
			SrcIP:     "127.0.0.1", // Parse from data
			DstIP:     "0.0.0.0",   // Parse from data
			SrcPort:   binary.LittleEndian.Uint16(data[16:18]),
			DstPort:   binary.LittleEndian.Uint16(data[18:20]),
			Bytes:     int64(binary.LittleEndian.Uint32(data[20:24])),
			Blocked:   true, // Network should be blocked in sandbox
		}

		e.networkIO = append(e.networkIO, event)

	case 3: // File event
		operation := "read"
		if len(data) > 24 && data[24] == 1 {
			operation = "write"
		}

		event := FileEvent{
			Timestamp: timestamp,
			Path:      "/unknown", // Parse from data
			Operation: operation,
			Mode:      binary.LittleEndian.Uint32(data[20:24]),
			Success:   data[24] == 0,
		}

		e.fileAccess = append(e.fileAccess, event)
	}
}

func (e *eBPFMonitor) GetSyscalls() []SyscallEvent {
	e.mu.RLock()
	defer e.mu.RUnlock()

	result := make([]SyscallEvent, len(e.syscalls))
	copy(result, e.syscalls)
	return result
}

func (e *eBPFMonitor) GetNetworkEvents() []NetworkEvent {
	e.mu.RLock()
	defer e.mu.RUnlock()

	result := make([]NetworkEvent, len(e.networkIO))
	copy(result, e.networkIO)
	return result
}

func (e *eBPFMonitor) GetFileEvents() []FileEvent {
	e.mu.RLock()
	defer e.mu.RUnlock()

	result := make([]FileEvent, len(e.fileAccess))
	copy(result, e.fileAccess)
	return result
}

// AddMonitoredPID inserts a PID into the monitored_pids map so that only these
// processes will be observed by the eBPF programs. Best-effort; returns error if map missing.
func (e *eBPFMonitor) AddMonitoredPID(pid uint32) error {
	if e.collection == nil {
		return fmt.Errorf("collection not initialized")
	}
	m := e.collection.Maps["monitored_pids"]
	if m == nil {
		return fmt.Errorf("monitored_pids map not found")
	}
	one := uint8(1)
	return m.Update(&pid, &one, ebpf.UpdateAny)
}

// initDangerousSyscalls fills the dangerous_syscalls map with a curated set of
// syscall numbers defined in dangerousSyscalls. Best-effort.
func (e *eBPFMonitor) initDangerousSyscalls() error {
	if e.collection == nil {
		return fmt.Errorf("collection not initialized")
	}
	m := e.collection.Maps["dangerous_syscalls"]
	if m == nil {
		return fmt.Errorf("dangerous_syscalls map not found")
	}
	for nr := range dangerousSyscalls {
		k := uint32(nr)
		v := uint8(1)
		_ = m.Update(&k, &v, ebpf.UpdateAny)
	}
	return nil
}
