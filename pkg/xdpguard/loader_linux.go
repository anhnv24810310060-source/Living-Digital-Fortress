//go:build linux

package xdpguard

import (
	"errors"
	"fmt"
	"net"
	"os"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
)

// Loader represents an XDP program loader/manager
type Loader interface {
	Attach(iface string, objPath string) error
	Detach(iface string) error
	UpdateMap(name string, key, value []byte) error
}

type loader struct {
	lnk  link.Link
	maps map[string]*ebpf.Map
}

// New returns a Linux XDP loader
func New() (Loader, error) { return &loader{maps: map[string]*ebpf.Map{}}, nil }

func (l *loader) Attach(iface string, objPath string) error {
	if objPath == "" {
		return errors.New("xdp object path required")
	}
	f, err := os.Open(objPath)
	if err != nil {
		return err
	}
	defer f.Close()
	spec, err := ebpf.LoadCollectionSpecFromReader(f)
	if err != nil {
		return fmt.Errorf("load spec: %w", err)
	}
	coll, err := ebpf.NewCollection(spec)
	if err != nil {
		return fmt.Errorf("new collection: %w", err)
	}
	// Choose program
	progName := os.Getenv("XDP_PROG")
	if progName == "" {
		// attempt common names
		if _, ok := coll.Programs["xdp_prog"]; ok {
			progName = "xdp_prog"
		} else {
			progName = "xdp_guard"
		}
	}
	prog, ok := coll.Programs[progName]
	if !ok {
		return fmt.Errorf("program not found in obj: %s", progName)
	}
	// Attach to iface
	ifi, err := net.InterfaceByName(iface)
	if err != nil {
		return err
	}
	lnk, err := link.AttachXDP(link.XDPOptions{Program: prog, Interface: ifi.Index})
	if err != nil {
		return fmt.Errorf("attach xdp: %w", err)
	}
	l.lnk = lnk
	// Keep map refs
	for name, m := range coll.Maps {
		l.maps[name] = m
	}
	return nil
}
func (l *loader) Detach(iface string) error {
	if l.lnk != nil {
		_ = l.lnk.Close()
		l.lnk = nil
	}
	for _, m := range l.maps {
		m.Close()
	}
	l.maps = map[string]*ebpf.Map{}
	return nil
}
func (l *loader) UpdateMap(name string, key, value []byte) error {
	m, ok := l.maps[name]
	if !ok {
		return fmt.Errorf("map not found: %s", name)
	}
	return m.Update(key, value, ebpf.UpdateAny)
}
