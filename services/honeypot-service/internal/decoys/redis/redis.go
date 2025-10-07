package redis

import "shieldx/shared/logging"

// Server is a placeholder for a Redis protocol honeypot.
type Server struct { Addr string }

func New(addr string) *Server { return &Server{Addr: addr} }

func (s *Server) Start() error {
    logging.Infof("redis decoy placeholder listening on %s (no real server yet)", s.Addr)
    return nil
}
