//go:build !quic

package main

// startQUICServer is disabled unless built with -tags quic
func startQUICServer(addr string) error { return nil }
