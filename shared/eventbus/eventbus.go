package eventbus

import (
    "context"
    "sync"
)

// Event represents a generic cross-service message.
type Event struct {
    Type    string
    Source  string
    Payload any
}

// Publisher publishes events.
type Publisher interface {
    Publish(ctx context.Context, evt Event) error
}

// Subscriber receives events of certain types.
type Subscriber interface {
    Handle(ctx context.Context, evt Event)
    Topics() []string
}

// Bus is a minimal in-memory pub/sub bus for early modular integration.
type Bus struct {
    mu    sync.RWMutex
    subs  map[string][]Subscriber
    queue chan Event
    stop  chan struct{}
}

// NewBus constructs an in-memory Bus.
func NewBus(buffer int) *Bus {
    b := &Bus{
        subs:  make(map[string][]Subscriber),
        queue: make(chan Event, buffer),
        stop:  make(chan struct{}),
    }
    go b.loop()
    return b
}

// loop dispatches events.
func (b *Bus) loop() {
    for {
        select {
        case evt := <-b.queue:
            b.dispatch(evt)
        case <-b.stop:
            return
        }
    }
}

// Close stops the bus.
func (b *Bus) Close() { close(b.stop) }

// Register adds a subscriber.
func (b *Bus) Register(sub Subscriber) {
    b.mu.Lock()
    defer b.mu.Unlock()
    for _, t := range sub.Topics() {
        b.subs[t] = append(b.subs[t], sub)
    }
}

// Publish enqueues an event.
func (b *Bus) Publish(ctx context.Context, evt Event) error {
    select {
    case b.queue <- evt:
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}

func (b *Bus) dispatch(evt Event) {
    b.mu.RLock()
    subs := append([]Subscriber(nil), b.subs[evt.Type]...)
    b.mu.RUnlock()
    for _, s := range subs {
        go s.Handle(context.Background(), evt)
    }
}
