package ssh

import "shieldx/shared/logging"

// Server is a placeholder for an SSH honeypot.
type Server struct { Addr string }

func New(addr string) *Server { return &Server{Addr: addr} }

func (s *Server) Start() error {
    // Real implementation would negotiate SSH handshake and capture credentials.
    logging.Infof("ssh decoy placeholder listening on %s (no real server yet)", s.Addr)
    return nil
}
