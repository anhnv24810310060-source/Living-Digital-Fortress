package http

import (
    "net/http"
    "time"
    "shieldx/shared/logging"
    "shieldx/shared/eventbus"
)

// Server is a minimal placeholder HTTP honeypot.
type Server struct {
    Addr string
    Bus  *eventbus.Bus
}

func New(addr string, bus *eventbus.Bus) *Server { return &Server{Addr: addr, Bus: bus} }

func (s *Server) Start() error {
    mux := http.NewServeMux()
    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        logging.Infof("http decoy request path=%s remote=%s ua=%s", r.URL.Path, r.RemoteAddr, r.Header.Get("User-Agent"))
        if s.Bus != nil {
            _ = s.Bus.Publish(r.Context(), eventbus.Event{Type: "honeypot.request", Source: "http-decoy", Payload: map[string]any{
                "path": r.URL.Path,
                "ua": r.Header.Get("User-Agent"),
                "remote": r.RemoteAddr,
                "ts": time.Now().UTC(),
            }})
        }
        w.WriteHeader(200)
        w.Write([]byte("ok"))
    })
    go func() {
        if err := http.ListenAndServe(s.Addr, mux); err != nil {
            logging.Errorf("http decoy stopped: %v", err)
        }
    }()
    logging.Infof("http decoy listening on %s", s.Addr)
    return nil
}
