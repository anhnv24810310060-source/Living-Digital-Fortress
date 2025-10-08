package wch

import (
	"context"
	"crypto/tls"
	"fmt"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

// CamouflageEngine handles TLS fingerprint rotation and traffic obfuscation
type CamouflageEngine struct {
	profiles       []TLSProfile
	currentProfile int
	rotationPeriod time.Duration
	mu             sync.RWMutex
	lastRotation   time.Time
	ja3Rotator     *JA3Rotator
}

// TLSProfile represents a TLS fingerprint profile
type TLSProfile struct {
	Name         string
	CipherSuites []uint16
	Curves       []tls.CurveID
	Extensions   []uint16
	UserAgent    string
	Headers      map[string]string
}

// JA3Rotator manages JA3 fingerprint rotation
type JA3Rotator struct {
	mu           sync.RWMutex
	fingerprints []JA3Fingerprint
	current      int
	rotateEvery  int // Rotate every N requests
	requestCount int
}

// JA3Fingerprint represents a JA3 fingerprint
type JA3Fingerprint struct {
	Hash         string
	TLSVersion   uint16
	CipherSuites []uint16
	Extensions   []uint16
	Curves       []tls.CurveID
	PointFormats []uint8
}

// CamouflageConfig configuration for camouflage engine
type CamouflageConfig struct {
	RotationPeriod time.Duration
	EnableJA3      bool
	CustomProfiles []TLSProfile
}

// NewCamouflageEngine creates a new camouflage engine
func NewCamouflageEngine(config CamouflageConfig) *CamouflageEngine {
	if config.RotationPeriod == 0 {
		config.RotationPeriod = 5 * time.Minute
	}

	profiles := config.CustomProfiles
	if len(profiles) == 0 {
		profiles = getDefaultProfiles()
	}

	engine := &CamouflageEngine{
		profiles:       profiles,
		currentProfile: 0,
		rotationPeriod: config.RotationPeriod,
		lastRotation:   time.Now(),
	}

	if config.EnableJA3 {
		engine.ja3Rotator = NewJA3Rotator(100) // Rotate every 100 requests
	}

	// Start rotation goroutine
	go engine.rotateLoop()

	return engine
}

// ApplyFingerprint applies the current fingerprint to the response
func (ce *CamouflageEngine) ApplyFingerprint(w http.ResponseWriter, r *http.Request) {
	ce.mu.RLock()
	profile := ce.profiles[ce.currentProfile]
	ce.mu.RUnlock()

	// Apply custom headers
	for key, value := range profile.Headers {
		w.Header().Set(key, value)
	}

	// Add timing jitter to prevent timing analysis
	jitter := time.Duration(rand.Intn(50)) * time.Millisecond
	time.Sleep(jitter)
}

// GetCurrentProfile returns the current TLS profile
func (ce *CamouflageEngine) GetCurrentProfile() TLSProfile {
	ce.mu.RLock()
	defer ce.mu.RUnlock()
	return ce.profiles[ce.currentProfile]
}

// RotateProfile manually rotates to the next profile
func (ce *CamouflageEngine) RotateProfile() {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	ce.currentProfile = (ce.currentProfile + 1) % len(ce.profiles)
	ce.lastRotation = time.Now()
}

// rotateLoop automatically rotates profiles
func (ce *CamouflageEngine) rotateLoop() {
	ticker := time.NewTicker(ce.rotationPeriod)
	defer ticker.Stop()

	for range ticker.C {
		ce.RotateProfile()
	}
}

// GetTLSConfig returns a TLS config for the current profile
func (ce *CamouflageEngine) GetTLSConfig() *tls.Config {
	profile := ce.GetCurrentProfile()

	return &tls.Config{
		CipherSuites:       profile.CipherSuites,
		CurvePreferences:   profile.Curves,
		MinVersion:         tls.VersionTLS12,
		MaxVersion:         tls.VersionTLS13,
		InsecureSkipVerify: false,
	}
}

// NewJA3Rotator creates a new JA3 rotator
func NewJA3Rotator(rotateEvery int) *JA3Rotator {
	return &JA3Rotator{
		fingerprints: getDefaultJA3Fingerprints(),
		current:      0,
		rotateEvery:  rotateEvery,
		requestCount: 0,
	}
}

// GetFingerprint returns the current JA3 fingerprint
func (jr *JA3Rotator) GetFingerprint() JA3Fingerprint {
	jr.mu.RLock()
	defer jr.mu.RUnlock()
	return jr.fingerprints[jr.current]
}

// Rotate rotates to the next fingerprint if needed
func (jr *JA3Rotator) Rotate() {
	jr.mu.Lock()
	defer jr.mu.Unlock()

	jr.requestCount++
	if jr.requestCount >= jr.rotateEvery {
		jr.current = (jr.current + 1) % len(jr.fingerprints)
		jr.requestCount = 0
	}
}

// getDefaultProfiles returns default TLS profiles mimicking common browsers
func getDefaultProfiles() []TLSProfile {
	return []TLSProfile{
		{
			Name: "Chrome",
			CipherSuites: []uint16{
				tls.TLS_AES_128_GCM_SHA256,
				tls.TLS_AES_256_GCM_SHA384,
				tls.TLS_CHACHA20_POLY1305_SHA256,
				tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
				tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			},
			Curves: []tls.CurveID{
				tls.X25519,
				tls.CurveP256,
				tls.CurveP384,
			},
			UserAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
			Headers: map[string]string{
				"Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
				"Accept-Language":           "en-US,en;q=0.9",
				"Accept-Encoding":           "gzip, deflate, br",
				"DNT":                       "1",
				"Connection":                "keep-alive",
				"Upgrade-Insecure-Requests": "1",
			},
		},
		{
			Name: "Firefox",
			CipherSuites: []uint16{
				tls.TLS_AES_128_GCM_SHA256,
				tls.TLS_CHACHA20_POLY1305_SHA256,
				tls.TLS_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
				tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			},
			Curves: []tls.CurveID{
				tls.X25519,
				tls.CurveP256,
				tls.CurveP384,
				tls.CurveP521,
			},
			UserAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
			Headers: map[string]string{
				"Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
				"Accept-Language":           "en-US,en;q=0.5",
				"Accept-Encoding":           "gzip, deflate, br",
				"DNT":                       "1",
				"Connection":                "keep-alive",
				"Upgrade-Insecure-Requests": "1",
			},
		},
		{
			Name: "Safari",
			CipherSuites: []uint16{
				tls.TLS_AES_128_GCM_SHA256,
				tls.TLS_AES_256_GCM_SHA384,
				tls.TLS_CHACHA20_POLY1305_SHA256,
				tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
			},
			Curves: []tls.CurveID{
				tls.X25519,
				tls.CurveP256,
				tls.CurveP384,
				tls.CurveP521,
			},
			UserAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
			Headers: map[string]string{
				"Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
				"Accept-Language": "en-US,en;q=0.9",
				"Accept-Encoding": "gzip, deflate, br",
				"Connection":      "keep-alive",
			},
		},
		{
			Name: "Edge",
			CipherSuites: []uint16{
				tls.TLS_AES_128_GCM_SHA256,
				tls.TLS_AES_256_GCM_SHA384,
				tls.TLS_CHACHA20_POLY1305_SHA256,
				tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
				tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			},
			Curves: []tls.CurveID{
				tls.X25519,
				tls.CurveP256,
				tls.CurveP384,
			},
			UserAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
			Headers: map[string]string{
				"Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
				"Accept-Language": "en-US,en;q=0.9",
				"Accept-Encoding": "gzip, deflate, br",
				"DNT":             "1",
				"Connection":      "keep-alive",
			},
		},
	}
}

// getDefaultJA3Fingerprints returns default JA3 fingerprints
func getDefaultJA3Fingerprints() []JA3Fingerprint {
	return []JA3Fingerprint{
		{
			Hash:       "771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24,0",
			TLSVersion: tls.VersionTLS13,
			CipherSuites: []uint16{
				tls.TLS_AES_128_GCM_SHA256,
				tls.TLS_AES_256_GCM_SHA384,
				tls.TLS_CHACHA20_POLY1305_SHA256,
			},
			Extensions: []uint16{0, 23, 65281, 10, 11, 35, 16, 5, 13, 18, 51, 45, 43, 27, 21},
			Curves: []tls.CurveID{
				tls.X25519,
				tls.CurveP256,
				tls.CurveP384,
			},
			PointFormats: []uint8{0},
		},
		{
			Hash:       "771,4865-4867-4866-49195-49199-52393-52392-49196-49200-49162-49161-49171-49172-51-57-47-53-10,0-23-65281-10-11-35-16-5-51-43-13-45-28-21,29-23-24-25-256-257,0",
			TLSVersion: tls.VersionTLS13,
			CipherSuites: []uint16{
				tls.TLS_AES_128_GCM_SHA256,
				tls.TLS_CHACHA20_POLY1305_SHA256,
				tls.TLS_AES_256_GCM_SHA384,
			},
			Extensions: []uint16{0, 23, 65281, 10, 11, 35, 16, 5, 51, 43, 13, 45, 28, 21},
			Curves: []tls.CurveID{
				tls.X25519,
				tls.CurveP256,
				tls.CurveP384,
				tls.CurveP521,
			},
			PointFormats: []uint8{0},
		},
	}
}

// ObfuscateTraffic adds random padding to make traffic analysis harder
func ObfuscateTraffic(data []byte, minPadding, maxPadding int) []byte {
	if maxPadding <= minPadding {
		return data
	}

	paddingSize := minPadding + rand.Intn(maxPadding-minPadding)
	padding := make([]byte, paddingSize)
	rand.Read(padding)

	// Append padding
	result := make([]byte, len(data)+paddingSize+4)
	copy(result[:4], []byte{0xDE, 0xAD, 0xBE, 0xEF}) // Magic bytes
	copy(result[4:4+len(data)], data)
	copy(result[4+len(data):], padding)

	return result
}

// DeobfuscateTraffic removes padding from obfuscated traffic
func DeobfuscateTraffic(data []byte) ([]byte, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("data too short")
	}

	// Check magic bytes
	if data[0] != 0xDE || data[1] != 0xAD || data[2] != 0xBE || data[3] != 0xEF {
		return data, nil // Not obfuscated
	}

	// Find original data length (before padding)
	// In real implementation, you'd store length in the header
	return data[4:], nil
}

// MimicHTTPTraffic wraps data to look like HTTP traffic
func MimicHTTPTraffic(data []byte) []byte {
	// Create fake HTTP request
	header := fmt.Sprintf("POST /api/v1/data HTTP/1.1\r\n"+
		"Host: cdn.example.com\r\n"+
		"User-Agent: Mozilla/5.0\r\n"+
		"Content-Type: application/octet-stream\r\n"+
		"Content-Length: %d\r\n"+
		"\r\n", len(data))

	result := append([]byte(header), data...)
	return result
}

// TimingObfuscation adds random delays to prevent timing analysis
func TimingObfuscation(ctx context.Context, minDelay, maxDelay time.Duration) {
	if maxDelay <= minDelay {
		return
	}

	delay := minDelay + time.Duration(rand.Int63n(int64(maxDelay-minDelay)))

	select {
	case <-ctx.Done():
		return
	case <-time.After(delay):
		return
	}
}
