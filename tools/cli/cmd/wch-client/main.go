package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"context"
	"crypto/ecdh"
	"crypto/tls"
	"encoding/hex"
	"shieldx/pkg/wch"

	quic "github.com/quic-go/quic-go"
)

type Config struct {
	Locator     string   `json:"locator"`
	Ingress     string   `json:"ingress"`
	IngressList []string `json:"ingress_list"`
	UseQUIC     bool     `json:"use_quic"`
	QuicAddr    string   `json:"quic_addr"`
	Tenant      string   `json:"tenant"`
	Scope       string   `json:"scope"`
	Path        string   `json:"path"`
	UDPTarget   string   `json:"udp_target"`
	UDPHex      string   `json:"udp_hex"`
	RebindMs    int      `json:"rebind_ms"`
}

func loadConfig(path string) *Config {
	if path == "" {
		return nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil
	}
	return &c
}

func main() {
	locator := flag.String("locator", "http://localhost:8080", "locator base URL")
	ingress := flag.String("ingress", "http://localhost:8081", "ingress base URL")
	useQUIC := flag.Bool("quic", false, "use QUIC stream/datagram to send sealed envelope (requires INGRESS_QUIC_ADDR)")
	quicAddr := flag.String("quic-addr", "localhost:9443", "QUIC server address (host:port)")
	pingSec := flag.Int("quic-ping", 10, "QUIC ping interval seconds (0=off)")
	idleSec := flag.Int("quic-idle", 60, "QUIC max idle timeout seconds")
	kaSec := flag.Int("quic-keepalive", 15, "QUIC keepalive seconds")
	tenant := flag.String("tenant", "demo", "tenant")
	scope := flag.String("scope", "api", "scope")
	path := flag.String("path", "/", "request path to guardian")
	udpTarget := flag.String("udp-target", "", "send sealed UDP to host:port via guardian/MASQUE")
	udpHex := flag.String("udp-hex", "", "hex payload for UDP (optional)")
	rebindMs := flag.Int("rebind-ms", 0, "periodically reconnect/rekey every N ms (0=off)")
	cfgPath := flag.String("config", "", "path to JSON config")
	flag.Parse()

	if c := loadConfig(*cfgPath); c != nil {
		if c.Locator != "" {
			*locator = c.Locator
		}
		if c.Ingress != "" {
			*ingress = c.Ingress
		}
		if c.UseQUIC {
			*useQUIC = true
		}
		if c.QuicAddr != "" {
			*quicAddr = c.QuicAddr
		}
		if c.Tenant != "" {
			*tenant = c.Tenant
		}
		if c.Scope != "" {
			*scope = c.Scope
		}
		if c.Path != "" {
			*path = c.Path
		}
		if c.UDPTarget != "" {
			*udpTarget = c.UDPTarget
		}
		if c.UDPHex != "" {
			*udpHex = c.UDPHex
		}
		if c.RebindMs > 0 {
			*rebindMs = c.RebindMs
		}
		// Prefer first ingress in list if provided
		if len(c.IngressList) > 0 {
			*ingress = c.IngressList[0]
		}
	}

	var qconn quic.Connection
	var qconf = &quic.Config{EnableDatagrams: true, KeepAlivePeriod: time.Duration(*kaSec) * time.Second, MaxIdleTimeout: time.Duration(*idleSec) * time.Second, TokenStore: quic.NewLRUTokenStore(64, 8)}

	// Background pinger to keep QUIC path alive and help migration
	pingStop := make(chan struct{})
	startPing := func() {
		if !*useQUIC {
			return
		}
		if *pingSec <= 0 {
			return
		}
		go func() {
			t := time.NewTicker(time.Duration(*pingSec) * time.Second)
			defer t.Stop()
			for {
				select {
				case <-pingStop:
					return
				case <-t.C:
					if qconn == nil {
						continue
					}
					// Use a short stream ping instead of datagram for compatibility
					st, err := qconn.OpenStreamSync(context.Background())
					if err != nil {
						continue
					}
					_, _ = st.Write([]byte(`{"ping":true}`))
					_ = st.Close()
				}
			}
		}()
	}

	if *useQUIC {
		tlsConf := &tls.Config{InsecureSkipVerify: true, NextProtos: []string{"shieldx-wch"}}
		c, err := quic.DialAddrEarly(context.Background(), *quicAddr, tlsConf, qconf)
		if err == nil {
			qconn = c
		}
		defer func() {
			if qconn != nil {
				qconn.CloseWithError(0, "bye")
			}
		}()
		startPing()
	}

	ensureQUICConn := func() error {
		if !*useQUIC {
			return nil
		}
		if qconn != nil {
			return nil
		}
		tlsConf := &tls.Config{InsecureSkipVerify: true, NextProtos: []string{"shieldx-wch"}}
		var lastErr error
		for i := 0; i < 5; i++ {
			c, err := quic.DialAddrEarly(context.Background(), *quicAddr, tlsConf, qconf)
			if err == nil {
				qconn = c
				return nil
			}
			lastErr = err
			time.Sleep(time.Duration(200*(i+1)) * time.Millisecond)
		}
		return lastErr
	}

	// One token + one channel; reuse ephemeral key; rekey via incrementing rekeyCounter
	// 1) Issue token once
	issueBody := map[string]any{"tenant": *tenant, "scope": *scope, "ttl_seconds": 600}
	tokResp := postJSONMust(*locator+"/issue", issueBody)
	token := tokResp["token"].(string)

	// 2) Connect once to get channel + guardian pubkey (try ingress list for failover)
	var connResp map[string]any
	ingresses := []string{*ingress}
	if c := loadConfig(*cfgPath); c != nil && len(c.IngressList) > 0 {
		ingresses = c.IngressList
	}
	for _, base := range ingresses {
		connResp = postJSONMay(base+"/connect", map[string]any{"token": token})
		if connResp != nil {
			*ingress = base
			break
		}
	}
	if connResp == nil {
		log.Fatalf("connect failed on all ingresses")
	}
	channelID := connResp["channelId"].(string)
	guardianPubB64 := connResp["guardianPubKey"].(string)
	if *rebindMs == 0 {
		if v, ok := connResp["rebindHintMs"]; ok {
			if f, ok2 := v.(float64); ok2 {
				*rebindMs = int(f)
			}
		}
	}

	// 3) Prepare ECDH once and reuse ephemeral pubkey
	curve := ecdh.X25519()
	clientPriv, clientPub, err := wch.GenerateClientEphemeral()
	if err != nil {
		panic(err)
	}
	guardianPub, err := wch.UnmarshalB64(guardianPubB64)
	if err != nil {
		panic(err)
	}
	gp, err := curve.NewPublicKey(guardianPub)
	if err != nil {
		panic(err)
	}
	shared, err := clientPriv.ECDH(gp)
	rekeyCounter := 0

	for {
		// 4) Build inner JSON (HTTP or UDP)
		var innerJSON []byte
		if *udpTarget != "" {
			data := []byte("ping")
			if *udpHex != "" {
				data = mustHex(*udpHex)
			}
			innerJSON = wch.ToJSON(wch.InnerUDPRequest{Target: *udpTarget, Data: data, TimeoutMs: 1000})
		} else {
			innerJSON = wch.ToJSON(wch.InnerRequest{Method: "GET", Path: *path})
		}

		// 5) Derive key for current counter and seal
		var key []byte
		if rekeyCounter > 0 {
			key, err = wch.DeriveKeyWithCounter(shared, channelID, rekeyCounter)
		} else {
			key, err = wch.DeriveKey(shared, channelID)
		}
		if err != nil {
			panic(err)
		}
		nonce, ct, err := wch.Seal(key, innerJSON)
		if err != nil {
			panic(err)
		}

		// 6) Send envelope using chosen transport
		env := map[string]any{
			"channelId":       channelID,
			"ephemeralPubKey": wch.MarshalB64(clientPub),
			"nonce":           wch.MarshalB64(nonce),
			"ciphertext":      wch.MarshalB64(ct),
			"rekeyCounter":    rekeyCounter,
		}
		var nonce2b64, ct2b64 string
		if *udpTarget != "" {
			sealedResp := postJSONMust(*ingress+"/wch/send-udp", env)
			nonce2b64 = sealedResp["nonce"].(string)
			ct2b64 = sealedResp["ciphertext"].(string)
		} else if *useQUIC {
			b, _ := json.Marshal(env)
			if qconn != nil || ensureQUICConn() == nil {
				nr, cr, err := sendQUICOnConn(qconn, b)
				if err != nil {
					log.Printf("datagram send failed: %v, retrying...", err)
					if qconn != nil {
						qconn.CloseWithError(0, "retry")
						qconn = nil
					}
					if ensureQUICConn() == nil {
						nr, cr, err = sendQUICOnConn(qconn, b)
					}
				}
				if err != nil {
					log.Printf("retry failed, fallback stream")
					nr, cr, err = sendQUIC(*quicAddr, b)
					if err != nil {
						log.Fatalf("QUIC send: %v", err)
					}
				}
				nonce2b64 = nr
				ct2b64 = cr
			} else {
				nr, cr, err := sendQUIC(*quicAddr, b)
				if err != nil {
					log.Fatalf("QUIC send: %v", err)
				}
				nonce2b64 = nr
				ct2b64 = cr
			}
		} else {
			sealedResp := postJSONMust(*ingress+"/wch/send", env)
			nonce2b64 = sealedResp["nonce"].(string)
			ct2b64 = sealedResp["ciphertext"].(string)
		}

		// 7) Open response with current key (server tolerates N/N-1); client attempts N-1 if needed
		nonce2, _ := wch.UnmarshalB64(nonce2b64)
		ct2, _ := wch.UnmarshalB64(ct2b64)
		pt2, err := wch.Open(key, nonce2, ct2)
		if err != nil && rekeyCounter > 0 {
			if keyPrev, err2 := wch.DeriveKeyWithCounter(shared, channelID, rekeyCounter-1); err2 == nil {
				pt2, err = wch.Open(keyPrev, nonce2, ct2)
			}
		}
		if err != nil {
			panic(err)
		}
		if *udpTarget == "" {
			var innerResp wch.InnerResponse
			if err := json.Unmarshal(pt2, &innerResp); err != nil {
				panic(err)
			}
			fmt.Printf("Status: %d\n", innerResp.Status)
			fmt.Printf("Body  : %s\n", string(innerResp.Body))
		} else {
			var udpResp wch.UDPResponse
			if err := json.Unmarshal(pt2, &udpResp); err != nil {
				panic(err)
			}
			fmt.Printf("UDP   : %d bytes\n", len(udpResp.Data))
		}

		if *rebindMs <= 0 {
			break
		}
		rekeyCounter++
		time.Sleep(time.Duration(*rebindMs) * time.Millisecond)
	}
}

func postJSONMust(url string, payload map[string]any) map[string]any {
	b, _ := json.Marshal(payload)
	client := &http.Client{Timeout: 10 * time.Second}
	res, err := client.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		panic(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		io.Copy(os.Stderr, res.Body)
		panic(fmt.Sprintf("HTTP %d", res.StatusCode))
	}
	var m map[string]any
	if err := json.NewDecoder(res.Body).Decode(&m); err != nil {
		panic(err)
	}
	return m
}

func postJSONMay(url string, payload map[string]any) map[string]any {
	b, _ := json.Marshal(payload)
	client := &http.Client{Timeout: 5 * time.Second}
	res, err := client.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		return nil
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil
	}
	var m map[string]any
	if err := json.NewDecoder(res.Body).Decode(&m); err != nil {
		return nil
	}
	return m
}

// sendQUIC opens a stream to the INGRESS_QUIC_ADDR and exchanges one request/response of sealed envelope.
func sendQUIC(addr string, body []byte) (nonceB64 string, ctB64 string, err error) {
	// Use quic-go client
	tlsConf := &tls.Config{InsecureSkipVerify: true, NextProtos: []string{"shieldx-wch"}}
	conn, err := quic.DialAddr(context.Background(), addr, tlsConf, nil)
	if err != nil {
		return "", "", err
	}
	defer conn.CloseWithError(0, "bye")
	st, err := conn.OpenStreamSync(context.Background())
	if err != nil {
		return "", "", err
	}
	if _, err := st.Write(body); err != nil {
		return "", "", err
	}
	_ = st.Close()
	respBytes, err := io.ReadAll(st)
	if err != nil {
		return "", "", err
	}
	var m map[string]string
	if err := json.Unmarshal(respBytes, &m); err != nil {
		return "", "", err
	}
	return m["nonce"], m["ciphertext"], nil
}

func sendQUICOnConn(conn quic.Connection, body []byte) (nonceB64 string, ctB64 string, err error) {
	if conn == nil {
		return "", "", fmt.Errorf("no quic conn")
	}
	st, err := conn.OpenStreamSync(context.Background())
	if err != nil {
		return "", "", err
	}
	if _, err := st.Write(body); err != nil {
		return "", "", err
	}
	_ = st.Close()
	respBytes, err := io.ReadAll(st)
	if err != nil {
		return "", "", err
	}
	var m map[string]string
	if err := json.Unmarshal(respBytes, &m); err != nil {
		return "", "", err
	}
	return m["nonce"], m["ciphertext"], nil
}

func mustHex(s string) []byte { b, _ := hex.DecodeString(s); return b }
