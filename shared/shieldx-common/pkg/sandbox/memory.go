//go:build linux

package sandbox

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type MemorySnapshot struct {
	PID        int                    `json:"pid"`
	Timestamp  time.Time              `json:"timestamp"`
	Regions    []MemoryRegion         `json:"regions"`
	Heap       []byte                 `json:"heap,omitempty"`
	Stack      []byte                 `json:"stack,omitempty"`
	Registers  map[string]uint64      `json:"registers"`
	Suspicious []SuspiciousPattern    `json:"suspicious"`
}

type MemoryRegion struct {
	Start       uint64 `json:"start"`
	End         uint64 `json:"end"`
	Permissions string `json:"permissions"`
	Path        string `json:"path"`
	Size        uint64 `json:"size"`
	Executable  bool   `json:"executable"`
	Writable    bool   `json:"writable"`
}

type SuspiciousPattern struct {
	Type        string `json:"type"`        // shellcode, rop_gadget, heap_spray
	Address     uint64 `json:"address"`
	Size        int    `json:"size"`
	Confidence  float64 `json:"confidence"`
	Description string `json:"description"`
	Bytes       []byte `json:"bytes,omitempty"`
}

// Known shellcode patterns and ROP gadgets
var shellcodePatterns = [][]byte{
	{0x31, 0xc0, 0x50, 0x68},                   // xor eax,eax; push eax; push
	{0x6a, 0x0b, 0x58, 0x99},                   // push 0xb; pop eax; cdq (execve)
	{0x48, 0x31, 0xc0, 0x48, 0x31, 0xdb},       // xor rax,rax; xor rbx,rbx (x64)
	{0x90, 0x90, 0x90, 0x90},                   // NOP sled
	{0xcc, 0xcc, 0xcc, 0xcc},                   // INT3 breakpoints
}

var ropGadgets = [][]byte{
	{0x58, 0xc3},             // pop rax; ret
	{0x5f, 0xc3},             // pop rdi; ret  
	{0x5e, 0xc3},             // pop rsi; ret
	{0x5a, 0xc3},             // pop rdx; ret
	{0x48, 0x89, 0xe0, 0xc3}, // mov rax, rsp; ret
}

func CaptureMemory(pid int) (*MemorySnapshot, error) {
	snapshot := &MemorySnapshot{
		PID:       pid,
		Timestamp: time.Now(),
		Registers: make(map[string]uint64),
		Regions:   make([]MemoryRegion, 0),
		Suspicious: make([]SuspiciousPattern, 0),
	}

	// Parse memory maps
	regions, err := parseMemoryMaps(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to parse memory maps: %w", err)
	}
	snapshot.Regions = regions

	// Capture registers via ptrace
	if err := captureRegisters(pid, snapshot); err != nil {
		return nil, fmt.Errorf("failed to capture registers: %w", err)
	}

	// Capture heap and stack
	if err := captureHeapStack(pid, snapshot); err != nil {
		return nil, fmt.Errorf("failed to capture heap/stack: %w", err)
	}

	// Scan for suspicious patterns
	scanSuspiciousPatterns(pid, snapshot)

	return snapshot, nil
}

func parseMemoryMaps(pid int) ([]MemoryRegion, error) {
	file, err := os.Open(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var regions []MemoryRegion
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) < 5 {
			continue
		}

		// Parse address range
		addrRange := strings.Split(parts[0], "-")
		if len(addrRange) != 2 {
			continue
		}

		start, err := strconv.ParseUint(addrRange[0], 16, 64)
		if err != nil {
			continue
		}

		end, err := strconv.ParseUint(addrRange[1], 16, 64)
		if err != nil {
			continue
		}

		permissions := parts[1]
		path := ""
		if len(parts) >= 6 {
			path = parts[5]
		}

		region := MemoryRegion{
			Start:       start,
			End:         end,
			Permissions: permissions,
			Path:        path,
			Size:        end - start,
			Executable:  strings.Contains(permissions, "x"),
			Writable:    strings.Contains(permissions, "w"),
		}

		regions = append(regions, region)
	}

	return regions, scanner.Err()
}

func captureRegisters(pid int, snapshot *MemorySnapshot) error {
	// Attach to process
	if err := syscall.PtraceAttach(pid); err != nil {
		return err
	}
	defer syscall.PtraceDetach(pid)

	// Wait for process to stop
	var status syscall.WaitStatus
	if _, err := syscall.Wait4(pid, &status, 0, nil); err != nil {
		return err
	}

	// Get registers (x86_64 specific)
	var regs syscall.PtraceRegs
	if err := syscall.PtraceGetRegs(pid, &regs); err != nil {
		return err
	}

	// Store important registers
	snapshot.Registers["rax"] = regs.Rax
	snapshot.Registers["rbx"] = regs.Rbx
	snapshot.Registers["rcx"] = regs.Rcx
	snapshot.Registers["rdx"] = regs.Rdx
	snapshot.Registers["rsi"] = regs.Rsi
	snapshot.Registers["rdi"] = regs.Rdi
	snapshot.Registers["rsp"] = regs.Rsp
	snapshot.Registers["rbp"] = regs.Rbp
	snapshot.Registers["rip"] = regs.Rip
	snapshot.Registers["r8"] = regs.R8
	snapshot.Registers["r9"] = regs.R9
	snapshot.Registers["r10"] = regs.R10
	snapshot.Registers["r11"] = regs.R11
	snapshot.Registers["r12"] = regs.R12
	snapshot.Registers["r13"] = regs.R13
	snapshot.Registers["r14"] = regs.R14
	snapshot.Registers["r15"] = regs.R15

	return nil
}

func captureHeapStack(pid int, snapshot *MemorySnapshot) error {
	memFile, err := os.Open(fmt.Sprintf("/proc/%d/mem", pid))
	if err != nil {
		return err
	}
	defer memFile.Close()

	// Find heap and stack regions
	for _, region := range snapshot.Regions {
		if strings.Contains(region.Path, "[heap]") && len(snapshot.Heap) == 0 {
			// Capture first 64KB of heap
			size := minU64(region.Size, uint64(64*1024))
			heap := make([]byte, int(size))
			
			if _, err := memFile.ReadAt(heap, int64(region.Start)); err == nil {
				snapshot.Heap = heap
			}
		}
		
		if strings.Contains(region.Path, "[stack]") && len(snapshot.Stack) == 0 {
			// Capture last 32KB of stack
			size := minU64(region.Size, uint64(32*1024))
			stack := make([]byte, int(size))
			
			offset := int64(region.End - size)
			if _, err := memFile.ReadAt(stack, offset); err == nil {
				snapshot.Stack = stack
			}
		}
	}

	return nil
}

func scanSuspiciousPatterns(pid int, snapshot *MemorySnapshot) {
	memFile, err := os.Open(fmt.Sprintf("/proc/%d/mem", pid))
	if err != nil {
		return
	}
	defer memFile.Close()

	// Scan executable regions for shellcode and ROP gadgets
	for _, region := range snapshot.Regions {
		if !region.Executable {
			continue
		}

	// Read region data (limit to 1MB per region)
	size := minU64(region.Size, uint64(1024*1024))
	data := make([]byte, int(size))
		
		if _, err := memFile.ReadAt(data, int64(region.Start)); err != nil {
			continue
		}

		// Scan for shellcode patterns
		for _, pattern := range shellcodePatterns {
			for i := 0; i <= len(data)-len(pattern); i++ {
				if matchesPattern(data[i:i+len(pattern)], pattern) {
					suspicious := SuspiciousPattern{
						Type:        "shellcode",
						Address:     region.Start + uint64(i),
						Size:        len(pattern),
						Confidence:  0.8,
						Description: fmt.Sprintf("Potential shellcode pattern at 0x%x", region.Start+uint64(i)),
						Bytes:       data[i:i+len(pattern)],
					}
					snapshot.Suspicious = append(snapshot.Suspicious, suspicious)
				}
			}
		}

		// Scan for ROP gadgets
		for _, gadget := range ropGadgets {
			for i := 0; i <= len(data)-len(gadget); i++ {
				if matchesPattern(data[i:i+len(gadget)], gadget) {
					suspicious := SuspiciousPattern{
						Type:        "rop_gadget",
						Address:     region.Start + uint64(i),
						Size:        len(gadget),
						Confidence:  0.7,
						Description: fmt.Sprintf("ROP gadget at 0x%x", region.Start+uint64(i)),
						Bytes:       data[i:i+len(gadget)],
					}
					snapshot.Suspicious = append(snapshot.Suspicious, suspicious)
				}
			}
		}

		// Detect NOP sleds (>= 16 consecutive NOPs)
		nopCount := 0
		for i, b := range data {
			if b == 0x90 {
				nopCount++
				if nopCount >= 16 {
					suspicious := SuspiciousPattern{
						Type:        "nop_sled",
						Address:     region.Start + uint64(i-nopCount+1),
						Size:        nopCount,
						Confidence:  0.9,
						Description: fmt.Sprintf("NOP sled (%d bytes) at 0x%x", nopCount, region.Start+uint64(i-nopCount+1)),
					}
					snapshot.Suspicious = append(snapshot.Suspicious, suspicious)
					nopCount = 0
				}
			} else {
				nopCount = 0
			}
		}
	}

	// Scan heap for spray patterns
	if len(snapshot.Heap) > 0 {
		detectHeapSpray(snapshot)
	}
}

func matchesPattern(data, pattern []byte) bool {
	if len(data) < len(pattern) {
		return false
	}
	
	for i, b := range pattern {
		if data[i] != b {
			return false
		}
	}
	return true
}

func detectHeapSpray(snapshot *MemorySnapshot) {
	heap := snapshot.Heap
	if len(heap) < 1024 {
		return
	}

	// Look for repeated 4-byte patterns (common in heap sprays)
	patternCounts := make(map[uint32]int)
	
	for i := 0; i <= len(heap)-4; i += 4 {
		pattern := binary.LittleEndian.Uint32(heap[i : i+4])
		patternCounts[pattern]++
	}

	// If any 4-byte pattern repeats more than 100 times, it's suspicious
	for pattern, count := range patternCounts {
		if count > 100 {
			suspicious := SuspiciousPattern{
				Type:        "heap_spray",
				Address:     0, // Heap base address would be added
				Size:        count * 4,
				Confidence:  0.85,
				Description: fmt.Sprintf("Heap spray pattern 0x%08x repeated %d times", pattern, count),
			}
			snapshot.Suspicious = append(snapshot.Suspicious, suspicious)
		}
	}
}

func minU64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}