package ml

import (
	"math"
	"testing"
)

func TestNewNetworkFlowExtractor(t *testing.T) {
	extractor := NewNetworkFlowExtractor()
	if extractor == nil {
		t.Fatal("NewNetworkFlowExtractor returned nil")
	}
}

func TestExtractFlowFeatures_EmptyPackets(t *testing.T) {
	extractor := NewNetworkFlowExtractor()
	features := extractor.ExtractFlowFeatures([]FlowPacket{})
	
	if features.PacketCount != 0 {
		t.Errorf("Expected PacketCount=0, got %d", features.PacketCount)
	}
	if features.TotalBytes != 0 {
		t.Errorf("Expected TotalBytes=0, got %d", features.TotalBytes)
	}
}

func TestExtractFlowFeatures_SinglePacket(t *testing.T) {
	extractor := NewNetworkFlowExtractor()
	
	packets := []FlowPacket{
		{
			Timestamp:  1000.0,
			Size:       100,
			Protocol:   "tcp",
			TCPFlags:   0x02, // SYN
			Payload:    []byte("hello"),
			Direction:  "forward",
			SourcePort: 12345,
			DestPort:   80,
		},
	}
	
	features := extractor.ExtractFlowFeatures(packets)
	
	if features.PacketCount != 1 {
		t.Errorf("Expected PacketCount=1, got %d", features.PacketCount)
	}
	if features.TotalBytes != 100 {
		t.Errorf("Expected TotalBytes=100, got %d", features.TotalBytes)
	}
	if features.SourcePort != 12345 {
		t.Errorf("Expected SourcePort=12345, got %d", features.SourcePort)
	}
	if features.DestPort != 80 {
		t.Errorf("Expected DestPort=80, got %d", features.DestPort)
	}
	if !features.IsWellKnownPort {
		t.Error("Port 80 should be classified as well-known")
	}
	if features.TCPSynCount != 1 {
		t.Errorf("Expected TCPSynCount=1, got %d", features.TCPSynCount)
	}
}

func TestExtractFlowFeatures_MultiplePackets(t *testing.T) {
	extractor := NewNetworkFlowExtractor()
	
	packets := []FlowPacket{
		{Timestamp: 1000.0, Size: 100, Protocol: "tcp", TCPFlags: 0x02, Direction: "forward"},  // SYN
		{Timestamp: 1000.1, Size: 150, Protocol: "tcp", TCPFlags: 0x12, Direction: "backward"}, // SYN+ACK
		{Timestamp: 1000.2, Size: 120, Protocol: "tcp", TCPFlags: 0x10, Direction: "forward"},  // ACK
		{Timestamp: 1000.5, Size: 200, Protocol: "tcp", TCPFlags: 0x18, Direction: "forward"},  // PSH+ACK
		{Timestamp: 1001.0, Size: 180, Protocol: "tcp", TCPFlags: 0x11, Direction: "backward"}, // FIN+ACK
	}
	
	features := extractor.ExtractFlowFeatures(packets)
	
	// Basic metrics
	if features.PacketCount != 5 {
		t.Errorf("Expected PacketCount=5, got %d", features.PacketCount)
	}
	expectedBytes := 100 + 150 + 120 + 200 + 180
	if features.TotalBytes != expectedBytes {
		t.Errorf("Expected TotalBytes=%d, got %d", expectedBytes, features.TotalBytes)
	}
	
	// Duration
	expectedDuration := 1001.0 - 1000.0
	if math.Abs(features.Duration-expectedDuration) > 0.001 {
		t.Errorf("Expected Duration=%.3f, got %.3f", expectedDuration, features.Duration)
	}
	
	// Rate metrics
	if features.PacketsPerSecond <= 0 {
		t.Error("PacketsPerSecond should be positive")
	}
	if features.BytesPerSecond <= 0 {
		t.Error("BytesPerSecond should be positive")
	}
	
	// TCP flags
	if features.TCPSynCount != 2 {
		t.Errorf("Expected TCPSynCount=2, got %d", features.TCPSynCount)
	}
	if features.TCPAckCount != 4 {
		t.Errorf("Expected TCPAckCount=4, got %d", features.TCPAckCount)
	}
	if features.TCPFinCount != 1 {
		t.Errorf("Expected TCPFinCount=1, got %d", features.TCPFinCount)
	}
	if features.TCPPshCount != 1 {
		t.Errorf("Expected TCPPshCount=1, got %d", features.TCPPshCount)
	}
	
	// Direction
	if features.ForwardRatio != 0.6 {
		t.Errorf("Expected ForwardRatio=0.6, got %.2f", features.ForwardRatio)
	}
	if features.BackwardRatio != 0.4 {
		t.Errorf("Expected BackwardRatio=0.4, got %.2f", features.BackwardRatio)
	}
}

func TestExtractFlowFeatures_SizeStatistics(t *testing.T) {
	extractor := NewNetworkFlowExtractor()
	
	packets := []FlowPacket{
		{Size: 100}, {Size: 200}, {Size: 150}, {Size: 300}, {Size: 250},
	}
	
	features := extractor.ExtractFlowFeatures(packets)
	
	// Mean: (100+200+150+300+250)/5 = 200
	if math.Abs(features.SizeMean-200.0) > 0.1 {
		t.Errorf("Expected SizeMean=200, got %.2f", features.SizeMean)
	}
	
	// Min and Max
	if features.SizeMin != 100 {
		t.Errorf("Expected SizeMin=100, got %.2f", features.SizeMin)
	}
	if features.SizeMax != 300 {
		t.Errorf("Expected SizeMax=300, got %.2f", features.SizeMax)
	}
	
	// Median (sorted: 100, 150, 200, 250, 300)
	if features.SizeMedian != 200 {
		t.Errorf("Expected SizeMedian=200, got %.2f", features.SizeMedian)
	}
	
	// Std should be positive
	if features.SizeStd <= 0 {
		t.Error("SizeStd should be positive")
	}
}

func TestExtractFlowFeatures_IATStatistics(t *testing.T) {
	extractor := NewNetworkFlowExtractor()
	
	packets := []FlowPacket{
		{Timestamp: 1.000}, // t=0
		{Timestamp: 1.010}, // IAT = 10ms
		{Timestamp: 1.020}, // IAT = 10ms
		{Timestamp: 1.040}, // IAT = 20ms
		{Timestamp: 1.050}, // IAT = 10ms
	}
	
	features := extractor.ExtractFlowFeatures(packets)
	
	// Mean IAT: (10+10+20+10)/4 = 12.5 ms
	expectedMean := 12.5
	if math.Abs(features.IATMean-expectedMean) > 0.1 {
		t.Errorf("Expected IATMean=%.2f, got %.2f", expectedMean, features.IATMean)
	}
	
	// Min and Max
	if math.Abs(features.IATMin-10.0) > 0.1 {
		t.Errorf("Expected IATMin=10, got %.2f", features.IATMin)
	}
	if math.Abs(features.IATMax-20.0) > 0.1 {
		t.Errorf("Expected IATMax=20, got %.2f", features.IATMax)
	}
}

func TestExtractFlowFeatures_ProtocolDiversity(t *testing.T) {
	extractor := NewNetworkFlowExtractor()
	
	packets := []FlowPacket{
		{Protocol: "tcp"},
		{Protocol: "tcp"},
		{Protocol: "udp"},
		{Protocol: "tcp"},
		{Protocol: "icmp"},
	}
	
	features := extractor.ExtractFlowFeatures(packets)
	
	// 3 protocols: tcp, udp, icmp
	if features.ProtocolDiversity != 3 {
		t.Errorf("Expected ProtocolDiversity=3, got %d", features.ProtocolDiversity)
	}
	
	// Protocol ratios
	if features.ProtocolRatios["tcp"] != 0.6 {
		t.Errorf("Expected tcp ratio=0.6, got %.2f", features.ProtocolRatios["tcp"])
	}
	if features.ProtocolRatios["udp"] != 0.2 {
		t.Errorf("Expected udp ratio=0.2, got %.2f", features.ProtocolRatios["udp"])
	}
	if features.ProtocolRatios["icmp"] != 0.2 {
		t.Errorf("Expected icmp ratio=0.2, got %.2f", features.ProtocolRatios["icmp"])
	}
}

func TestExtractFlowFeatures_PayloadEntropy(t *testing.T) {
	extractor := NewNetworkFlowExtractor()
	
	// Test with random-looking data (high entropy)
	packets := []FlowPacket{
		{Payload: []byte{0x00, 0xFF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC}},
		{Payload: []byte{0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66}},
	}
	
	features := extractor.ExtractFlowFeatures(packets)
	
	// Entropy should be positive
	if features.PayloadEntropy <= 0 {
		t.Error("PayloadEntropy should be positive for varied data")
	}
	
	// Test with uniform data (low entropy)
	packets2 := []FlowPacket{
		{Payload: []byte{0x00, 0x00, 0x00, 0x00}},
		{Payload: []byte{0x00, 0x00, 0x00, 0x00}},
	}
	
	features2 := extractor.ExtractFlowFeatures(packets2)
	
	// Should be 0 entropy (all same byte)
	if features2.PayloadEntropy != 0 {
		t.Errorf("Expected PayloadEntropy=0 for uniform data, got %.2f", features2.PayloadEntropy)
	}
}

func TestExtractFlowFeatures_PortClassification(t *testing.T) {
	tests := []struct {
		name             string
		port             int
		wantWellKnown    bool
		wantRegistered   bool
		wantDynamic      bool
	}{
		{"HTTP", 80, true, false, false},
		{"HTTPS", 443, true, false, false},
		{"Custom app", 8080, false, true, false},
		{"High port", 3000, false, true, false},
		{"Dynamic port", 50000, false, false, true},
		{"Max port", 65535, false, false, true},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := NewNetworkFlowExtractor()
			packets := []FlowPacket{{DestPort: tt.port}}
			features := extractor.ExtractFlowFeatures(packets)
			
			if features.IsWellKnownPort != tt.wantWellKnown {
				t.Errorf("IsWellKnownPort = %v, want %v", features.IsWellKnownPort, tt.wantWellKnown)
			}
			if features.IsRegisteredPort != tt.wantRegistered {
				t.Errorf("IsRegisteredPort = %v, want %v", features.IsRegisteredPort, tt.wantRegistered)
			}
			if features.IsDynamicPort != tt.wantDynamic {
				t.Errorf("IsDynamicPort = %v, want %v", features.IsDynamicPort, tt.wantDynamic)
			}
		})
	}
}

func TestNetworkFlowFeatures_ToVector(t *testing.T) {
	features := &NetworkFlowFeatures{
		PacketCount:      10,
		TotalBytes:       1000,
		Duration:         1.5,
		PacketsPerSecond: 6.67,
		IsWellKnownPort:  true,
		IsRegisteredPort: false,
		IsDynamicPort:    false,
	}
	
	vector := features.ToVector()
	
	// Should have 27 features
	if len(vector) != 27 {
		t.Errorf("Expected vector length 27, got %d", len(vector))
	}
	
	// Check some values
	if vector[0] != 10 { // PacketCount
		t.Errorf("Expected vector[0]=10, got %.2f", vector[0])
	}
	if vector[1] != 1000 { // TotalBytes
		t.Errorf("Expected vector[1]=1000, got %.2f", vector[1])
	}
	if vector[24] != 1.0 { // IsWellKnownPort (true)
		t.Errorf("Expected vector[24]=1.0, got %.2f", vector[24])
	}
	if vector[25] != 0.0 { // IsRegisteredPort (false)
		t.Errorf("Expected vector[25]=0.0, got %.2f", vector[25])
	}
}

func TestCalculateMean(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{"empty", []float64{}, 0},
		{"single", []float64{5.0}, 5.0},
		{"multiple", []float64{1, 2, 3, 4, 5}, 3.0},
		{"negative", []float64{-1, -2, -3}, -2.0},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateMean(tt.values)
			if math.Abs(got-tt.want) > 0.001 {
				t.Errorf("calculateMean() = %.3f, want %.3f", got, tt.want)
			}
		})
	}
}

func TestCalculateStdDev(t *testing.T) {
	values := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	mean := calculateMean(values)
	stddev := calculateStdDev(values, mean)
	
	// Expected stddev ≈ 2.0
	expected := 2.0
	if math.Abs(stddev-expected) > 0.1 {
		t.Errorf("calculateStdDev() = %.3f, want %.3f", stddev, expected)
	}
}

func TestCalculateMedian(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{"empty", []float64{}, 0},
		{"single", []float64{5.0}, 5.0},
		{"odd count", []float64{1, 3, 5}, 3.0},
		{"even count", []float64{1, 2, 3, 4}, 2.5},
		{"unsorted", []float64{5, 1, 3, 2, 4}, 3.0},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateMedian(tt.values)
			if math.Abs(got-tt.want) > 0.001 {
				t.Errorf("calculateMedian() = %.3f, want %.3f", got, tt.want)
			}
		})
	}
}

func TestCalculateDiscreteEntropy(t *testing.T) {
	tests := []struct {
		name   string
		values []int
		want   float64
	}{
		{"empty", []int{}, 0},
		{"uniform", []int{1, 1, 1, 1}, 0},
		{"binary equal", []int{0, 1, 0, 1}, 1.0},
		{"varied", []int{1, 2, 3, 4, 5, 6, 7, 8}, 3.0},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateDiscreteEntropy(tt.values)
			if math.Abs(got-tt.want) > 0.1 {
				t.Errorf("calculateDiscreteEntropy() = %.3f, want %.3f", got, tt.want)
			}
		})
	}
}

func TestCalculateByteArrayEntropy(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want float64
	}{
		{"empty", []byte{}, 0},
		{"uniform", []byte{0x00, 0x00, 0x00, 0x00}, 0},
		{"binary", []byte{0x00, 0xFF, 0x00, 0xFF}, 1.0},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateByteArrayEntropy(tt.data)
			if math.Abs(got-tt.want) > 0.1 {
				t.Errorf("calculateByteArrayEntropy() = %.3f, want %.3f", got, tt.want)
			}
		})
	}
}

func BenchmarkExtractFlowFeatures(b *testing.B) {
	extractor := NewNetworkFlowExtractor()
	
	// Create realistic flow with 100 packets
	packets := make([]FlowPacket, 100)
	for i := 0; i < 100; i++ {
		packets[i] = FlowPacket{
			Timestamp:  float64(i) * 0.01,
			Size:       100 + i*10,
			Protocol:   "tcp",
			TCPFlags:   0x10,
			Payload:    make([]byte, 50),
			Direction:  "forward",
			SourcePort: 12345,
			DestPort:   80,
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor.ExtractFlowFeatures(packets)
	}
}

func BenchmarkToVector(b *testing.B) {
	features := &NetworkFlowFeatures{
		PacketCount:       100,
		TotalBytes:        10000,
		Duration:          1.5,
		PacketsPerSecond:  66.67,
		BytesPerSecond:    6666.67,
		SizeMean:          100,
		ProtocolDiversity: 2,
		ProtocolRatios:    map[string]float64{"tcp": 0.8, "udp": 0.2},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		features.ToVector()
	}
}
