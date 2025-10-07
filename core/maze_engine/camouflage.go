package maze_engine

import (
	crand "crypto/rand"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"shieldx/shared/ledger"
	"shieldx/shared/metrics"
)

type CamouflageEngine struct {
	templates    map[string]*Template
	sessions     map[string]*Session
	mu           sync.RWMutex
	templatePath string
	metrics      *CamouflageMetrics
}

type Template struct {
	Name                    string                 `json:"name"`
	Version                 string                 `json:"version"`
	FingerprintID           string                 `json:"fingerprint_id"`
	Headers                 map[string]string      `json:"headers"`
	ErrorPages              map[string]ErrorPage   `json:"error_pages"`
	BehavioralPatterns      BehavioralPatterns     `json:"behavioral_patterns"`
	VulnerabilitySimulation map[string]interface{} `json:"vulnerability_simulation"`
	SSLConfig               SSLConfig              `json:"ssl_config"`
	Modules                 []string               `json:"modules"`
	LogFormat               string                 `json:"log_format"`
	ReconnaissanceResponses map[string]interface{} `json:"reconnaissance_responses"`
}

type ErrorPage struct {
	Title       string `json:"title"`
	Body        string `json:"body"`
	ContentType string `json:"content_type"`
}

type BehavioralPatterns struct {
	ResponseTiming ResponseTiming `json:"response_timing"`
	KeepAlive      KeepAlive      `json:"keep_alive"`
	Compression    Compression    `json:"compression"`
}

type ResponseTiming struct {
	MinMs        int     `json:"min_ms"`
	MaxMs        int     `json:"max_ms"`
	Distribution string  `json:"distribution"`
	JitterFactor float64 `json:"jitter_factor"`
}

type KeepAlive struct {
	Timeout     int `json:"timeout"`
	MaxRequests int `json:"max_requests"`
}

type Compression struct {
	Gzip    bool `json:"gzip"`
	Deflate bool `json:"deflate"`
	Br      bool `json:"br"`
	MinSize int  `json:"min_size"`
}

type SSLConfig struct {
	Protocols       []string               `json:"protocols"`
	Ciphers         string                 `json:"ciphers"`
	CertificateInfo map[string]interface{} `json:"certificate_info"`
	HSTS            map[string]interface{} `json:"hsts"`
}

type Session struct {
	ID           string
	Template     *Template
	StartTime    time.Time
	RequestCount int
	ClientIP     string
	UserAgent    string
	Fingerprint  string
	LastActivity time.Time
}

type CamouflageMetrics struct {
	TemplateRequests  *metrics.Counter
	SessionsActive    *metrics.Gauge
	EngagementRate    *metrics.Counter
	FingerprintMisses *metrics.Counter
	ResponseLatency   *metrics.Histogram
}

func NewCamouflageEngine(templatePath string) (*CamouflageEngine, error) {
	engine := &CamouflageEngine{
		templates:    make(map[string]*Template),
		sessions:     make(map[string]*Session),
		templatePath: templatePath,
		metrics:      initMetrics(),
	}

	if err := engine.loadTemplates(); err != nil {
		return nil, fmt.Errorf("failed to load templates: %w", err)
	}

	// Start cleanup goroutine
	go engine.sessionCleanup()

	return engine, nil
}

func initMetrics() *CamouflageMetrics {
	return &CamouflageMetrics{
		TemplateRequests:  metrics.NewCounter("camouflage_template_requests_total", "Total template requests"),
		SessionsActive:    metrics.NewGauge("camouflage_sessions_active", "Active camouflage sessions"),
		EngagementRate:    metrics.NewCounter("camouflage_engagement_total", "Total reconnaissance engagements"),
		FingerprintMisses: metrics.NewCounter("camouflage_fingerprint_misses_total", "Fingerprint detection misses"),
		ResponseLatency:   metrics.NewHistogram("camouflage_response_duration_seconds", "Response latency", nil),
	}
}

func (ce *CamouflageEngine) loadTemplates() error {
	return filepath.WalkDir(ce.templatePath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !strings.HasSuffix(path, ".json") {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read template %s: %w", path, err)
		}

		var template Template
		if err := json.Unmarshal(data, &template); err != nil {
			return fmt.Errorf("failed to parse template %s: %w", path, err)
		}

		ce.mu.Lock()
		ce.templates[template.Name] = &template
		ce.mu.Unlock()

		return nil
	})
}

func (ce *CamouflageEngine) GetTemplate(name string) (*Template, error) {
	ce.mu.RLock()
	template, exists := ce.templates[name]
	ce.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("template %s not found", name)
	}

	ce.metrics.TemplateRequests.Inc()
	return template, nil
}

func (ce *CamouflageEngine) ListTemplates() []string {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	names := make([]string, 0, len(ce.templates))
	for name := range ce.templates {
		names = append(names, name)
	}
	return names
}

func (ce *CamouflageEngine) CreateSession(templateName, clientIP, userAgent string) (*Session, error) {
	template, err := ce.GetTemplate(templateName)
	if err != nil {
		return nil, err
	}

	sessionID := generateSessionID()
	fingerprint := generateFingerprint(clientIP, userAgent)

	session := &Session{
		ID:           sessionID,
		Template:     template,
		StartTime:    time.Now(),
		RequestCount: 0,
		ClientIP:     clientIP,
		UserAgent:    userAgent,
		Fingerprint:  fingerprint,
		LastActivity: time.Now(),
	}

	ce.mu.Lock()
	ce.sessions[sessionID] = session
	ce.mu.Unlock()

	// Update active sessions gauge by counting sessions
	ce.mu.RLock()
	active := len(ce.sessions)
	ce.mu.RUnlock()
	ce.metrics.SessionsActive.Set(uint64(active))

	// Audit log
	_ = ledger.AppendJSONLine("data/ledger-camouflage.log", "camouflage", "session_created", map[string]any{
		"session_id":  sessionID,
		"template":    templateName,
		"client_ip":   clientIP,
		"fingerprint": fingerprint,
	})

	return session, nil
}

func (ce *CamouflageEngine) GetSession(sessionID string) (*Session, bool) {
	ce.mu.RLock()
	session, exists := ce.sessions[sessionID]
	ce.mu.RUnlock()

	if exists {
		session.LastActivity = time.Now()
		session.RequestCount++
	}

	return session, exists
}

func (ce *CamouflageEngine) ApplyTemplate(w http.ResponseWriter, r *http.Request, session *Session) {
	start := time.Now()
	defer func() {
		ce.metrics.ResponseLatency.Observe(time.Since(start).Seconds())
	}()

	template := session.Template

	// Apply headers
	for key, value := range template.Headers {
		w.Header().Set(key, ce.interpolateVariables(value, r, session))
	}

	// Apply behavioral timing
	delay := ce.calculateResponseDelay(template.BehavioralPatterns.ResponseTiming)
	time.Sleep(delay)

	// Handle specific paths
	statusCode, body := ce.handleRequest(r, template)

	w.WriteHeader(statusCode)
	w.Write([]byte(ce.interpolateVariables(body, r, session)))

	// Log engagement
	ce.metrics.EngagementRate.Inc()

	_ = ledger.AppendJSONLine("data/ledger-camouflage.log", "camouflage", "request_handled", map[string]any{
		"session_id":  session.ID,
		"path":        r.URL.Path,
		"method":      r.Method,
		"status_code": statusCode,
		"user_agent":  r.UserAgent(),
		"client_ip":   session.ClientIP,
	})
}

func (ce *CamouflageEngine) handleRequest(r *http.Request, template *Template) (int, string) {
	path := r.URL.Path

	// Check for vulnerability simulation
	if ce.isVulnerabilityProbe(path, template) {
		ce.metrics.FingerprintMisses.Inc()
		return ce.handleVulnerabilityProbe(path, template)
	}

	// Handle error pages
	switch {
	case strings.Contains(path, "admin"):
		if errorPage, exists := template.ErrorPages["403"]; exists {
			return 403, errorPage.Body
		}
		return 403, "Forbidden"
	case strings.Contains(path, "nonexistent"):
		if errorPage, exists := template.ErrorPages["404"]; exists {
			return 404, errorPage.Body
		}
		return 404, "Not Found"
	default:
		return 200, ce.generateDefaultPage(template)
	}
}

func (ce *CamouflageEngine) isVulnerabilityProbe(path string, template *Template) bool {
	vulnSim := template.VulnerabilitySimulation
	if vulnSim == nil {
		return false
	}

	// Check for common vulnerability patterns
	patterns := []string{"../", "..\\", "%2e%2e", "/etc/passwd", "/admin", "/.git", "/backup", "database.sql"}
	for _, pattern := range patterns {
		if strings.Contains(strings.ToLower(path), pattern) {
			return true
		}
	}

	// Check template-specific vulnerability paths
	for _, vuln := range vulnSim {
		if vulnMap, ok := vuln.(map[string]interface{}); ok {
			if paths, exists := vulnMap["paths"]; exists {
				if pathList, ok := paths.([]interface{}); ok {
					for _, p := range pathList {
						if pathStr, ok := p.(string); ok && strings.Contains(path, pathStr) {
							return true
						}
					}
				}
			}
		}
	}

	return false
}

func (ce *CamouflageEngine) handleVulnerabilityProbe(path string, template *Template) (int, string) {
	// Return appropriate response based on template configuration
	if errorPage, exists := template.ErrorPages["403"]; exists {
		return 403, errorPage.Body
	}
	return 400, "Bad Request: Potential vulnerability detected"
}

func (ce *CamouflageEngine) generateDefaultPage(template *Template) string {
	switch template.Name {
	case "apache":
		return `<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<head><title>Index of /</title></head>
<body>
<h1>Index of /</h1>
<table>
<tr><th valign="top"><img src="/icons/blank.gif" alt="[ICO]"></th><th><a href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th><th><a href="?C=S;O=A">Size</a></th><th><a href="?C=D;O=A">Description</a></th></tr>
<tr><th colspan="5"><hr></th></tr>
</table>
<address>Apache/2.4.54 (Ubuntu) Server</address>
</body></html>`
	case "nginx":
		return `<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
</head>
<body>
<h1>Welcome to nginx!</h1>
<p>If you see this page, the nginx web server is successfully installed and working.</p>
<p>For online documentation and support please refer to <a href="http://nginx.org/">nginx.org</a>.</p>
</body>
</html>`
	case "iis":
		return `<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>IIS Windows Server</title>
</head>
<body>
<div id="header"><h1>Internet Information Services</h1></div>
<div id="content">
<h2>Welcome</h2>
<p>This is the default web site for this server. The web site content is located in the default directory.</p>
</div>
</body>
</html>`
	default:
		serverBanner := template.Headers["Server"]
		if serverBanner == "" {
			serverBanner = "Generic/1.0"
		}
		return fmt.Sprintf("<html><body><h1>Welcome</h1><p>Served by %s</p></body></html>", serverBanner)
	}
}

func (ce *CamouflageEngine) calculateResponseDelay(timing ResponseTiming) time.Duration {
	var delay float64

	switch timing.Distribution {
	case "normal":
		// Normal distribution
		mean := float64(timing.MinMs+timing.MaxMs) / 2
		stddev := float64(timing.MaxMs-timing.MinMs) / 6
		delay = normalRandom(mean, stddev)
	case "exponential":
		// Exponential distribution
		lambda := 1.0 / float64(timing.MinMs)
		delay = exponentialRandom(lambda)
	default:
		// Uniform distribution
		delay = float64(timing.MinMs) + rand.Float64()*float64(timing.MaxMs-timing.MinMs)
	}

	// Apply jitter
	jitter := 1.0 + (rand.Float64()-0.5)*timing.JitterFactor
	delay *= jitter

	// Ensure within bounds
	if delay < float64(timing.MinMs) {
		delay = float64(timing.MinMs)
	}
	if delay > float64(timing.MaxMs) {
		delay = float64(timing.MaxMs)
	}

	return time.Duration(delay) * time.Millisecond
}

func (ce *CamouflageEngine) interpolateVariables(text string, r *http.Request, session *Session) string {
	replacements := map[string]string{
		"{{host}}":      r.Host,
		"{{path}}":      r.URL.Path,
		"{{port}}":      "80",
		"{{client_ip}}": session.ClientIP,
		"{{timestamp}}": time.Now().Format("02/Jan/2006:15:04:05 -0700"),
		"{{pid}}":       "12345",
		"{{tid}}":       "67890",
	}

	result := text
	for placeholder, value := range replacements {
		result = strings.ReplaceAll(result, placeholder, value)
	}

	return result
}

func (ce *CamouflageEngine) sessionCleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		ce.mu.Lock()
		now := time.Now()
		for id, session := range ce.sessions {
			if now.Sub(session.LastActivity) > 30*time.Minute {
				delete(ce.sessions, id)
			}
		}
		// refresh gauge after cleanup
		active := len(ce.sessions)
		ce.metrics.SessionsActive.Set(uint64(active))
		ce.mu.Unlock()
	}
}

func generateSessionID() string {
	bytes := make([]byte, 16)
	crand.Read(bytes)
	return fmt.Sprintf("%x", bytes)
}

func generateFingerprint(clientIP, userAgent string) string {
	data := fmt.Sprintf("%s:%s:%d", clientIP, userAgent, time.Now().Unix())
	bytes := make([]byte, 8)
	_ = data // currently unused; may be used for hashing in future
	crand.Read(bytes)
	return fmt.Sprintf("%x", bytes)
}

func normalRandom(mean, stddev float64) float64 {
	// Box-Muller transform
	u1 := rand.Float64()
	u2 := rand.Float64()
	z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	return mean + stddev*z0
}

func exponentialRandom(lambda float64) float64 {
	u := rand.Float64()
	return -math.Log(1-u) / lambda
}
