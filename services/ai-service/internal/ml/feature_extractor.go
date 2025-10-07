package ml

import (
	"math"
	"net"
	"time"
)

type FeatureExtractor struct {
	windowSize time.Duration
	buffer     []NetworkFlow
}

type NetworkFlow struct {
	Timestamp   time.Time
	SrcIP       net.IP
	DstIP       net.IP
	SrcPort     uint16
	DstPort     uint16
	Protocol    string
	PacketSize  int
	TCPFlags    uint8
	PayloadSize int
	PayloadHash [32]byte
}

type ConnectionMetrics struct {
	Duration          time.Duration
	PacketCount       int
	ByteCount         int64
	PacketSizes       []int
	InterArrivalTimes []time.Duration
	TCPFlags          []uint8
	PayloadEntropy    float64
	PortDiversity     int
	IPDiversity       int
}

func NewFeatureExtractor(windowSize time.Duration) *FeatureExtractor {
	return &FeatureExtractor{
		windowSize: windowSize,
		buffer:     make([]NetworkFlow, 0),
	}
}

func (fe *FeatureExtractor) AddFlow(flow NetworkFlow) {
	fe.buffer = append(fe.buffer, flow)

	// Clean old flows outside window
	cutoff := time.Now().Add(-fe.windowSize)
	for i, f := range fe.buffer {
		if f.Timestamp.After(cutoff) {
			fe.buffer = fe.buffer[i:]
			break
		}
	}
}

func (fe *FeatureExtractor) ExtractFeatures() []float64 {
	if len(fe.buffer) == 0 {
		return make([]float64, 18) // Return zero vector
	}

	metrics := fe.calculateMetrics()
	features := make([]float64, 18)

	// Packet size statistics
	features[0] = fe.calculateMean(metrics.PacketSizes)
	features[1] = fe.calculateStd(metrics.PacketSizes)
	features[2] = float64(metrics.PacketCount)

	// Connection duration and rates
	features[3] = metrics.Duration.Seconds()
	if metrics.Duration.Seconds() > 0 {
		features[4] = float64(metrics.ByteCount) / metrics.Duration.Seconds()   // bytes/sec
		features[5] = float64(metrics.PacketCount) / metrics.Duration.Seconds() // packets/sec
	}

	// TCP flags entropy
	features[6] = fe.calculateTCPFlagsEntropy(metrics.TCPFlags)

	// Payload entropy
	features[7] = metrics.PayloadEntropy

	// Inter-arrival time statistics
	features[8] = fe.calculateMeanDuration(metrics.InterArrivalTimes)
	features[9] = fe.calculateStdDuration(metrics.InterArrivalTimes)

	// Diversity metrics
	features[10] = float64(metrics.PortDiversity)
	features[11] = float64(metrics.IPDiversity)

	// Syscall-related features (from sandbox integration)
	features[12] = fe.getSyscallCount()
	features[13] = fe.getDangerousSyscallRatio()
	features[14] = fe.getMemoryAllocations()
	features[15] = fe.getFileOperations()
	features[16] = fe.getNetworkOperations()
	features[17] = fe.getProcessSpawns()

	return features
}

func (fe *FeatureExtractor) calculateMetrics() ConnectionMetrics {
	if len(fe.buffer) == 0 {
		return ConnectionMetrics{}
	}

	metrics := ConnectionMetrics{
		PacketSizes:       make([]int, 0, len(fe.buffer)),
		InterArrivalTimes: make([]time.Duration, 0, len(fe.buffer)-1),
		TCPFlags:          make([]uint8, 0, len(fe.buffer)),
	}

	// Calculate basic metrics
	var totalBytes int64
	srcPorts := make(map[uint16]bool)
	dstPorts := make(map[uint16]bool)
	srcIPs := make(map[string]bool)
	dstIPs := make(map[string]bool)

	var payloadEntropy float64
	var entropyCount int

	for i, flow := range fe.buffer {
		metrics.PacketSizes = append(metrics.PacketSizes, flow.PacketSize)
		totalBytes += int64(flow.PacketSize)

		if flow.Protocol == "tcp" {
			metrics.TCPFlags = append(metrics.TCPFlags, flow.TCPFlags)
		}

		// Calculate inter-arrival times
		if i > 0 {
			interArrival := flow.Timestamp.Sub(fe.buffer[i-1].Timestamp)
			metrics.InterArrivalTimes = append(metrics.InterArrivalTimes, interArrival)
		}

		// Track diversity
		srcPorts[flow.SrcPort] = true
		dstPorts[flow.DstPort] = true
		srcIPs[flow.SrcIP.String()] = true
		dstIPs[flow.DstIP.String()] = true

		// Calculate payload entropy
		if flow.PayloadSize > 0 {
			entropy := fe.calculateShannonEntropy(flow.PayloadHash[:])
			payloadEntropy += entropy
			entropyCount++
		}
	}

	metrics.PacketCount = len(fe.buffer)
	metrics.ByteCount = totalBytes
	metrics.Duration = fe.buffer[len(fe.buffer)-1].Timestamp.Sub(fe.buffer[0].Timestamp)
	metrics.PortDiversity = len(srcPorts) + len(dstPorts)
	metrics.IPDiversity = len(srcIPs) + len(dstIPs)

	if entropyCount > 0 {
		metrics.PayloadEntropy = payloadEntropy / float64(entropyCount)
	}

	return metrics
}

func (fe *FeatureExtractor) calculateMean(values []int) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0
	for _, v := range values {
		sum += v
	}
	return float64(sum) / float64(len(values))
}

func (fe *FeatureExtractor) calculateStd(values []int) float64 {
	if len(values) <= 1 {
		return 0
	}

	mean := fe.calculateMean(values)
	sumSquares := 0.0

	for _, v := range values {
		diff := float64(v) - mean
		sumSquares += diff * diff
	}

	return math.Sqrt(sumSquares / float64(len(values)-1))
}

func (fe *FeatureExtractor) calculateMeanDuration(durations []time.Duration) float64 {
	if len(durations) == 0 {
		return 0
	}

	sum := time.Duration(0)
	for _, d := range durations {
		sum += d
	}
	return float64(sum) / float64(len(durations)) / float64(time.Millisecond)
}

func (fe *FeatureExtractor) calculateStdDuration(durations []time.Duration) float64 {
	if len(durations) <= 1 {
		return 0
	}

	mean := fe.calculateMeanDuration(durations)
	sumSquares := 0.0

	for _, d := range durations {
		diff := float64(d)/float64(time.Millisecond) - mean
		sumSquares += diff * diff
	}

	return math.Sqrt(sumSquares / float64(len(durations)-1))
}

func (fe *FeatureExtractor) calculateTCPFlagsEntropy(flags []uint8) float64 {
	if len(flags) == 0 {
		return 0
	}

	// Count frequency of each flag combination
	flagCounts := make(map[uint8]int)
	for _, flag := range flags {
		flagCounts[flag]++
	}

	// Calculate Shannon entropy
	entropy := 0.0
	total := float64(len(flags))

	for _, count := range flagCounts {
		if count > 0 {
			p := float64(count) / total
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

func (fe *FeatureExtractor) calculateShannonEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0
	}

	// Count byte frequencies
	freq := make(map[byte]int)
	for _, b := range data {
		freq[b]++
	}

	// Calculate entropy
	entropy := 0.0
	total := float64(len(data))

	for _, count := range freq {
		if count > 0 {
			p := float64(count) / total
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

// Syscall-related features (integration with sandbox monitoring)
func (fe *FeatureExtractor) getSyscallCount() float64 {
	// This would integrate with the eBPF monitor
	// For now, return a placeholder
	return 0.0
}

func (fe *FeatureExtractor) getDangerousSyscallRatio() float64 {
	// Ratio of dangerous syscalls to total syscalls
	return 0.0
}

func (fe *FeatureExtractor) getMemoryAllocations() float64 {
	// Number of memory allocation syscalls (mmap, brk, etc.)
	return 0.0
}

func (fe *FeatureExtractor) getFileOperations() float64 {
	// Number of file I/O operations
	return 0.0
}

func (fe *FeatureExtractor) getNetworkOperations() float64 {
	// Number of network-related syscalls
	return 0.0
}

func (fe *FeatureExtractor) getProcessSpawns() float64 {
	// Number of process creation syscalls
	return 0.0
}

// Advanced feature extraction for specific attack patterns
func (fe *FeatureExtractor) ExtractAdvancedFeatures() map[string]float64 {
	features := make(map[string]float64)

	// Port scan detection
	features["port_scan_score"] = fe.detectPortScan()

	// DDoS patterns
	features["ddos_score"] = fe.detectDDoSPattern()

	// Tunneling behavior
	features["tunnel_score"] = fe.detectTunneling()

	// Covert channel detection
	features["covert_channel_score"] = fe.detectCovertChannel()

	// Timing attack patterns
	features["timing_attack_score"] = fe.detectTimingAttack()

	return features
}

func (fe *FeatureExtractor) detectPortScan() float64 {
	if len(fe.buffer) < 10 {
		return 0.0
	}

	// Count unique destination ports per source IP
	srcToPortCount := make(map[string]int)

	for _, flow := range fe.buffer {
		srcIP := flow.SrcIP.String()
		srcToPortCount[srcIP]++
	}

	// High port diversity from single source indicates scanning
	maxPorts := 0
	for _, count := range srcToPortCount {
		if count > maxPorts {
			maxPorts = count
		}
	}

	// Normalize to 0-1 scale
	return math.Min(float64(maxPorts)/100.0, 1.0)
}

func (fe *FeatureExtractor) detectDDoSPattern() float64 {
	if len(fe.buffer) < 50 {
		return 0.0
	}

	// High packet rate with low diversity indicates DDoS
	timeWindow := 10 * time.Second
	now := time.Now()

	recentCount := 0
	for _, flow := range fe.buffer {
		if now.Sub(flow.Timestamp) <= timeWindow {
			recentCount++
		}
	}

	packetsPerSecond := float64(recentCount) / timeWindow.Seconds()

	// Normalize: >1000 pps = high DDoS score
	return math.Min(packetsPerSecond/1000.0, 1.0)
}

func (fe *FeatureExtractor) detectTunneling() float64 {
	// Look for consistent payload sizes and timing (DNS tunneling, etc.)
	if len(fe.buffer) < 20 {
		return 0.0
	}

	// Check for DNS traffic with unusual patterns
	dnsFlows := 0
	for _, flow := range fe.buffer {
		if flow.DstPort == 53 || flow.SrcPort == 53 {
			dnsFlows++
		}
	}

	if dnsFlows == 0 {
		return 0.0
	}

	// High DNS traffic ratio indicates potential tunneling
	dnsRatio := float64(dnsFlows) / float64(len(fe.buffer))
	return math.Min(dnsRatio*2.0, 1.0)
}

func (fe *FeatureExtractor) detectCovertChannel() float64 {
	// Detect timing-based covert channels
	if len(fe.buffer) < 30 {
		return 0.0
	}

	// Calculate timing regularity
	intervals := make([]float64, 0, len(fe.buffer)-1)
	for i := 1; i < len(fe.buffer); i++ {
		interval := fe.buffer[i].Timestamp.Sub(fe.buffer[i-1].Timestamp)
		intervals = append(intervals, interval.Seconds())
	}

	// Low variance in timing indicates potential covert channel
	if len(intervals) == 0 {
		return 0.0
	}

	mean := 0.0
	for _, interval := range intervals {
		mean += interval
	}
	mean /= float64(len(intervals))

	variance := 0.0
	for _, interval := range intervals {
		diff := interval - mean
		variance += diff * diff
	}
	variance /= float64(len(intervals))

	// Low variance = high covert channel score
	if variance < 0.001 {
		return 1.0
	}
	return math.Max(0.0, 1.0-variance*1000)
}

func (fe *FeatureExtractor) detectTimingAttack() float64 {
	// Detect precise timing measurements (side-channel attacks)
	if len(fe.buffer) < 100 {
		return 0.0
	}

	// Look for very precise timing patterns
	preciseTimings := 0
	for i := 1; i < len(fe.buffer); i++ {
		interval := fe.buffer[i].Timestamp.Sub(fe.buffer[i-1].Timestamp)
		// Microsecond-level precision indicates timing attack
		if interval.Nanoseconds()%1000 == 0 && interval < time.Millisecond {
			preciseTimings++
		}
	}

	precisionRatio := float64(preciseTimings) / float64(len(fe.buffer)-1)
	return math.Min(precisionRatio*10.0, 1.0)
}
