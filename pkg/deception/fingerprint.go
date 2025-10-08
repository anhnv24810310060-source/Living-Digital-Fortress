package deception

import (
	"crypto/tls"
	"net/http"
	"time"
)

type FingerprintMimic struct {
	HTTPProfiles map[string]*HTTPProfile
	TLSProfiles  map[string]*TLSProfile
	SSHProfiles  map[string]*SSHProfile
}

type HTTPProfile struct {
	Server       string
	Headers      map[string]string
	ResponseTime time.Duration
	StatusCodes  []int
}

type TLSProfile struct {
	CipherSuites []uint16
	Curves       []tls.CurveID
	MinVersion   uint16
	MaxVersion   uint16
	ServerName   string
}

type SSHProfile struct {
	Banner      string
	Algorithms  []string
	AuthMethods []string
}

func NewFingerprintMimic() *FingerprintMimic {
	fm := &FingerprintMimic{
		HTTPProfiles: make(map[string]*HTTPProfile),
		TLSProfiles:  make(map[string]*TLSProfile),
		SSHProfiles:  make(map[string]*SSHProfile),
	}

	fm.loadDefaultProfiles()
	return fm
}

func (fm *FingerprintMimic) loadDefaultProfiles() {
	// Apache HTTP profile
	fm.HTTPProfiles["apache"] = &HTTPProfile{
		Server: "Apache/2.4.41 (Ubuntu)",
		Headers: map[string]string{
			"X-Powered-By": "PHP/7.4.3",
			"Connection":   "Keep-Alive",
			"Keep-Alive":   "timeout=5, max=100",
		},
		ResponseTime: 50 * time.Millisecond,
		StatusCodes:  []int{200, 404, 403, 500},
	}

	// Nginx HTTP profile
	fm.HTTPProfiles["nginx"] = &HTTPProfile{
		Server: "nginx/1.18.0",
		Headers: map[string]string{
			"Connection": "keep-alive",
			"Vary":       "Accept-Encoding",
		},
		ResponseTime: 30 * time.Millisecond,
		StatusCodes:  []int{200, 404, 502, 503},
	}

	// IIS HTTP profile
	fm.HTTPProfiles["iis"] = &HTTPProfile{
		Server: "Microsoft-IIS/10.0",
		Headers: map[string]string{
			"X-Powered-By":     "ASP.NET",
			"X-AspNet-Version": "4.0.30319",
		},
		ResponseTime: 80 * time.Millisecond,
		StatusCodes:  []int{200, 404, 500},
	}

	// Apache TLS profile
	fm.TLSProfiles["apache"] = &TLSProfile{
		MinVersion: tls.VersionTLS12,
		MaxVersion: tls.VersionTLS13,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_RSA_WITH_AES_256_GCM_SHA384,
		},
		Curves: []tls.CurveID{
			tls.CurveP256,
			tls.CurveP384,
		},
	}

	// Nginx TLS profile
	fm.TLSProfiles["nginx"] = &TLSProfile{
		MinVersion: tls.VersionTLS12,
		MaxVersion: tls.VersionTLS13,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		},
		Curves: []tls.CurveID{
			tls.X25519,
			tls.CurveP256,
		},
	}

	// OpenSSH profile
	fm.SSHProfiles["openssh"] = &SSHProfile{
		Banner: "SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5",
		Algorithms: []string{
			"diffie-hellman-group14-sha256",
			"ecdh-sha2-nistp256",
			"ecdh-sha2-nistp384",
		},
		AuthMethods: []string{"publickey", "password"},
	}

	// Dropbear SSH profile
	fm.SSHProfiles["dropbear"] = &SSHProfile{
		Banner: "SSH-2.0-dropbear_2020.81",
		Algorithms: []string{
			"diffie-hellman-group14-sha1",
			"ecdh-sha2-nistp256",
		},
		AuthMethods: []string{"password"},
	}
}

func (fm *FingerprintMimic) MimicHTTPServer(profile string) http.Handler {
	p := fm.HTTPProfiles[profile]
	if p == nil {
		p = fm.HTTPProfiles["apache"]
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Add server-specific headers
		w.Header().Set("Server", p.Server)
		for k, v := range p.Headers {
			w.Header().Set(k, v)
		}

		// Add timestamp header with slight jitter
		w.Header().Set("Date", time.Now().Format(http.TimeFormat))

		// Simulate response time
		time.Sleep(p.ResponseTime)

		// Serve different content based on path
		switch r.URL.Path {
		case "/admin":
			w.WriteHeader(http.StatusForbidden)
			w.Write([]byte("<html><body><h1>403 Forbidden</h1></body></html>"))
		case "/login":
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`<html><body>
				<form method="post">
					<input type="text" name="username" placeholder="Username">
					<input type="password" name="password" placeholder="Password">
					<input type="submit" value="Login">
				</form>
			</body></html>`))
		default:
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("<html><body><h1>Welcome</h1></body></html>"))
		}
	})
}

func (fm *FingerprintMimic) MimicTLSConfig(profile string) *tls.Config {
	p := fm.TLSProfiles[profile]
	if p == nil {
		// Default Apache-like TLS config
		return &tls.Config{
			MinVersion: tls.VersionTLS12,
			MaxVersion: tls.VersionTLS13,
			CipherSuites: []uint16{
				tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			},
		}
	}

	return &tls.Config{
		MinVersion:       p.MinVersion,
		MaxVersion:       p.MaxVersion,
		CipherSuites:     p.CipherSuites,
		CurvePreferences: p.Curves,
		ServerName:       p.ServerName,
	}
}

func (fm *FingerprintMimic) GetSSHBanner(profile string) string {
	p := fm.SSHProfiles[profile]
	if p == nil {
		return fm.SSHProfiles["openssh"].Banner
	}
	return p.Banner
}

func (fm *FingerprintMimic) GetSSHAlgorithms(profile string) []string {
	p := fm.SSHProfiles[profile]
	if p == nil {
		return fm.SSHProfiles["openssh"].Algorithms
	}
	return p.Algorithms
}

func (fm *FingerprintMimic) AddCustomProfile(name string, httpProfile *HTTPProfile, tlsProfile *TLSProfile, sshProfile *SSHProfile) {
	if httpProfile != nil {
		fm.HTTPProfiles[name] = httpProfile
	}
	if tlsProfile != nil {
		fm.TLSProfiles[name] = tlsProfile
	}
	if sshProfile != nil {
		fm.SSHProfiles[name] = sshProfile
	}
}
