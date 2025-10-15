package ml

import (
	"math"
	"sort"
)

// NetworkFlowFeatures represents comprehensive network flow features
type NetworkFlowFeatures struct {
	// Basic flow metrics
	PacketCount   int     `json:"packet_count"`
	TotalBytes    int     `json:"total_bytes"`
	Duration      float64 `json:"duration_seconds"`
	
	// Rate features
	PacketsPerSecond float64 `json:"packets_per_second"`
	BytesPerSecond   float64 `json:"bytes_per_second"`
	
	// Packet size statistics
	SizeMean      float64 `json:"size_mean"`
	SizeStd       float64 `json:"size_std"`
	SizeMin       float64 `json:"size_min"`
	SizeMax       float64 `json:"size_max"`
	SizeMedian    float64 `json:"size_median"`
	
	// Inter-arrival time statistics
	IATMean       float64 `json:"iat_mean_ms"`
	IATStd        float64 `json:"iat_std_ms"`
	IATMin        float64 `json:"iat_min_ms"`
	IATMax        float64 `json:"iat_max_ms"`
	
	// Protocol features
	ProtocolDiversity int                `json:"protocol_diversity"`
	ProtocolRatios    map[string]float64 `json:"protocol_ratios"`
	
	// TCP-specific features
	TCPSynCount     int     `json:"tcp_syn_count"`
	TCPAckCount     int     `json:"tcp_ack_count"`
	TCPFinCount     int     `json:"tcp_fin_count"`
	TCPRstCount     int     `json:"tcp_rst_count"`
	TCPPshCount     int     `json:"tcp_psh_count"`
	TCPFlagsEntropy float64 `json:"tcp_flags_entropy"`
	
	// Payload features
	PayloadEntropy float64 `json:"payload_entropy"`
	
	// Direction features
	ForwardRatio  float64 `json:"forward_ratio"`
	BackwardRatio float64 `json:"backward_ratio"`
	
	// Port features
	SourcePort       int  `json:"source_port"`
	DestPort         int  `json:"dest_port"`
	IsWellKnownPort  bool `json:"is_well_known_port"`
	IsRegisteredPort bool `json:"is_registered_port"`
	IsDynamicPort    bool `json:"is_dynamic_port"`
}

// FlowPacket represents a single packet in a flow
type FlowPacket struct {
	Timestamp  float64 `json:"timestamp"`   // Unix timestamp with milliseconds
	Size       int     `json:"size"`        // bytes
	Protocol   string  `json:"protocol"`    // tcp, udp, icmp
	TCPFlags   int     `json:"tcp_flags"`   // TCP flags bitmap
	Payload    []byte  `json:"payload"`     // packet payload
	Direction  string  `json:"direction"`   // "forward" or "backward"
	SourcePort int     `json:"source_port"`
	DestPort   int     `json:"dest_port"`
}

// NetworkFlowExtractor extracts features from network flows
type NetworkFlowExtractor struct{}

// NewNetworkFlowExtractor creates a new flow feature extractor
func NewNetworkFlowExtractor() *NetworkFlowExtractor {
	return &NetworkFlowExtractor{}
}

// ExtractFlowFeatures extracts features from a flow (slice of packets)
func (nfe *NetworkFlowExtractor) ExtractFlowFeatures(packets []FlowPacket) *NetworkFlowFeatures {
	if len(packets) == 0 {
		return &NetworkFlowFeatures{
			ProtocolRatios: make(map[string]float64),
		}
	}
	
	features := &NetworkFlowFeatures{
		PacketCount:    len(packets),
		ProtocolRatios: make(map[string]float64),
	}
	
	// Extract all feature categories
	nfe.extractBasicMetrics(packets, features)
	nfe.extractRateMetrics(packets, features)
	nfe.extractSizeStatistics(packets, features)
	nfe.extractIATStatistics(packets, features)
	nfe.extractProtocolFeatures(packets, features)
	nfe.extractTCPFeatures(packets, features)
	nfe.extractPayloadFeatures(packets, features)
	nfe.extractDirectionFeatures(packets, features)
	nfe.extractPortFeatures(packets, features)
	
	return features
}

func (nfe *NetworkFlowExtractor) extractBasicMetrics(packets []FlowPacket, features *NetworkFlowFeatures) {
	totalBytes := 0
	for _, pkt := range packets {
		totalBytes += pkt.Size
	}
	features.TotalBytes = totalBytes
	
	if len(packets) > 1 {
		features.Duration = packets[len(packets)-1].Timestamp - packets[0].Timestamp
	}
}

func (nfe *NetworkFlowExtractor) extractRateMetrics(packets []FlowPacket, features *NetworkFlowFeatures) {
	if features.Duration > 0 {
		features.PacketsPerSecond = float64(features.PacketCount) / features.Duration
		features.BytesPerSecond = float64(features.TotalBytes) / features.Duration
	}
}

func (nfe *NetworkFlowExtractor) extractSizeStatistics(packets []FlowPacket, features *NetworkFlowFeatures) {
	sizes := make([]float64, len(packets))
	for i, pkt := range packets {
		sizes[i] = float64(pkt.Size)
	}
	
	features.SizeMean = calculateMean(sizes)
	features.SizeStd = calculateStdDev(sizes, features.SizeMean)
	features.SizeMin = calculateMin(sizes)
	features.SizeMax = calculateMax(sizes)
	features.SizeMedian = calculateMedian(sizes)
}

func (nfe *NetworkFlowExtractor) extractIATStatistics(packets []FlowPacket, features *NetworkFlowFeatures) {
	if len(packets) < 2 {
		return
	}
	
	iats := make([]float64, len(packets)-1)
	for i := 1; i < len(packets); i++ {
		// Convert to milliseconds
		iats[i-1] = (packets[i].Timestamp - packets[i-1].Timestamp) * 1000
	}
	
	features.IATMean = calculateMean(iats)
	features.IATStd = calculateStdDev(iats, features.IATMean)
	features.IATMin = calculateMin(iats)
	features.IATMax = calculateMax(iats)
}

func (nfe *NetworkFlowExtractor) extractProtocolFeatures(packets []FlowPacket, features *NetworkFlowFeatures) {
	protocolCounts := make(map[string]int)
	
	for _, pkt := range packets {
		protocolCounts[pkt.Protocol]++
	}
	
	features.ProtocolDiversity = len(protocolCounts)
	
	for protocol, count := range protocolCounts {
		features.ProtocolRatios[protocol] = float64(count) / float64(len(packets))
	}
}

func (nfe *NetworkFlowExtractor) extractTCPFeatures(packets []FlowPacket, features *NetworkFlowFeatures) {
	flags := []int{}
	
	for _, pkt := range packets {
		if pkt.Protocol == "tcp" {
			flags = append(flags, pkt.TCPFlags)
			
			// Count TCP flags (standard bit positions)
			if pkt.TCPFlags&0x02 != 0 { features.TCPSynCount++ } // SYN
			if pkt.TCPFlags&0x10 != 0 { features.TCPAckCount++ } // ACK
			if pkt.TCPFlags&0x01 != 0 { features.TCPFinCount++ } // FIN
			if pkt.TCPFlags&0x04 != 0 { features.TCPRstCount++ } // RST
			if pkt.TCPFlags&0x08 != 0 { features.TCPPshCount++ } // PSH
		}
	}
	
	if len(flags) > 0 {
		features.TCPFlagsEntropy = calculateDiscreteEntropy(flags)
	}
}

func (nfe *NetworkFlowExtractor) extractPayloadFeatures(packets []FlowPacket, features *NetworkFlowFeatures) {
	allPayload := []byte{}
	
	for _, pkt := range packets {
		allPayload = append(allPayload, pkt.Payload...)
	}
	
	if len(allPayload) > 0 {
		features.PayloadEntropy = calculateByteArrayEntropy(allPayload)
	}
}

func (nfe *NetworkFlowExtractor) extractDirectionFeatures(packets []FlowPacket, features *NetworkFlowFeatures) {
	forwardCount := 0
	backwardCount := 0
	
	for _, pkt := range packets {
		if pkt.Direction == "forward" {
			forwardCount++
		} else if pkt.Direction == "backward" {
			backwardCount++
		}
	}
	
	if len(packets) > 0 {
		features.ForwardRatio = float64(forwardCount) / float64(len(packets))
		features.BackwardRatio = float64(backwardCount) / float64(len(packets))
	}
}

func (nfe *NetworkFlowExtractor) extractPortFeatures(packets []FlowPacket, features *NetworkFlowFeatures) {
	if len(packets) == 0 {
		return
	}
	
	features.SourcePort = packets[0].SourcePort
	features.DestPort = packets[0].DestPort
	
	// Port classification
	features.IsWellKnownPort = features.DestPort > 0 && features.DestPort < 1024
	features.IsRegisteredPort = features.DestPort >= 1024 && features.DestPort < 49152
	features.IsDynamicPort = features.DestPort >= 49152 && features.DestPort <= 65535
}

// ToVector converts features to ML-ready vector (27 features)
func (features *NetworkFlowFeatures) ToVector() []float64 {
	return []float64{
		float64(features.PacketCount),
		float64(features.TotalBytes),
		features.Duration,
		features.PacketsPerSecond,
		features.BytesPerSecond,
		features.SizeMean,
		features.SizeStd,
		features.SizeMin,
		features.SizeMax,
		features.SizeMedian,
		features.IATMean,
		features.IATStd,
		features.IATMin,
		features.IATMax,
		float64(features.ProtocolDiversity),
		float64(features.TCPSynCount),
		float64(features.TCPAckCount),
		float64(features.TCPFinCount),
		float64(features.TCPRstCount),
		float64(features.TCPPshCount),
		features.TCPFlagsEntropy,
		features.PayloadEntropy,
		features.ForwardRatio,
		features.BackwardRatio,
		boolToFloat64(features.IsWellKnownPort),
		boolToFloat64(features.IsRegisteredPort),
		boolToFloat64(features.IsDynamicPort),
	}
}

// Statistical utility functions

func calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateStdDev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	return math.Sqrt(variance / float64(len(values)))
}

func calculateMin(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	minVal := values[0]
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func calculateMax(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	maxVal := values[0]
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2.0
	}
	return sorted[mid]
}

func calculateDiscreteEntropy(values []int) float64 {
	if len(values) == 0 {
		return 0
	}
	
	counts := make(map[int]int)
	for _, v := range values {
		counts[v]++
	}
	
	entropy := 0.0
	total := float64(len(values))
	for _, count := range counts {
		prob := float64(count) / total
		if prob > 0 {
			entropy -= prob * math.Log2(prob)
		}
	}
	
	return entropy
}

func calculateByteArrayEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0
	}
	
	counts := make([]int, 256)
	for _, b := range data {
		counts[b]++
	}
	
	entropy := 0.0
	total := float64(len(data))
	for _, count := range counts {
		if count > 0 {
			prob := float64(count) / total
			entropy -= prob * math.Log2(prob)
		}
	}
	
	return entropy
}

func boolToFloat64(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
