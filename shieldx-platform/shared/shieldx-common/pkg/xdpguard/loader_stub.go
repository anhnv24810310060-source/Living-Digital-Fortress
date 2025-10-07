//go:build !linux

package xdpguard

type Loader interface {
    Attach(iface string, objPath string) error
    Detach(iface string) error
}

type loader struct{}

func New() (Loader, error) { return &loader{}, nil }
func (l *loader) Attach(iface string, objPath string) error { return nil }
func (l *loader) Detach(iface string) error { return nil }



