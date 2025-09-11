package orchestrator

import (
	"fmt"
	"sync"
	"time"
)

type EdgeFabric struct {
	pops     map[string]*PoP
	channels map[string]*ChannelPool
	router   *MeshRouter
	mu       sync.RWMutex
}

type PoP struct {
	ID       string
	Region   string
	Endpoint string
	Load     int
	Latency  time.Duration
	Capacity int
	Active   bool
}

type ChannelPool struct {
	TenantID    string
	PreWarmed   []*Channel
	Active      []*Channel
	MaxChannels int
	WarmupTime  time.Duration
}

type Channel struct {
	ID        string
	TenantID  string
	PopID     string
	CreatedAt time.Time
	LastUsed  time.Time
	Ready     bool
}

type MeshRouter struct {
	routes map[string]*Route
	mu     sync.RWMutex
}

type Route struct {
	Source      string
	Destination string
	Hops        []string
	Latency     time.Duration
}

func NewEdgeFabric() *EdgeFabric {
	return &EdgeFabric{
		pops:     make(map[string]*PoP),
		channels: make(map[string]*ChannelPool),
		router:   NewMeshRouter(),
	}
}

func (ef *EdgeFabric) AddPoP(id, region, endpoint string, capacity int) *PoP {
	ef.mu.Lock()
	defer ef.mu.Unlock()
	
	pop := &PoP{
		ID:       id,
		Region:   region,
		Endpoint: endpoint,
		Capacity: capacity,
		Active:   true,
		Load:     0,
	}
	
	ef.pops[id] = pop
	return pop
}

func (ef *EdgeFabric) CreateChannelPool(tenantID string, maxChannels int) *ChannelPool {
	ef.mu.Lock()
	defer ef.mu.Unlock()
	
	pool := &ChannelPool{
		TenantID:    tenantID,
		PreWarmed:   make([]*Channel, 0),
		Active:      make([]*Channel, 0),
		MaxChannels: maxChannels,
		WarmupTime:  100 * time.Millisecond,
	}
	
	ef.channels[tenantID] = pool
	go ef.warmupChannels(pool)
	
	return pool
}

func (ef *EdgeFabric) GetChannel(tenantID string) (*Channel, error) {
	ef.mu.Lock()
	defer ef.mu.Unlock()
	
	pool := ef.channels[tenantID]
	if pool == nil {
		pool = ef.CreateChannelPool(tenantID, 10)
	}
	
	if len(pool.PreWarmed) > 0 {
		channel := pool.PreWarmed[0]
		pool.PreWarmed = pool.PreWarmed[1:]
		pool.Active = append(pool.Active, channel)
		channel.LastUsed = time.Now()
		
		go ef.warmupChannels(pool)
		return channel, nil
	}
	
	if len(pool.Active) < pool.MaxChannels {
		channel := ef.createChannel(tenantID)
		pool.Active = append(pool.Active, channel)
		return channel, nil
	}
	
	return nil, fmt.Errorf("channel pool exhausted")
}

func (ef *EdgeFabric) MigrateChannel(channelID, newPopID string) error {
	ef.mu.Lock()
	defer ef.mu.Unlock()
	
	var channel *Channel
	for _, pool := range ef.channels {
		for _, ch := range pool.Active {
			if ch.ID == channelID {
				channel = ch
				break
			}
		}
	}
	
	if channel == nil {
		return fmt.Errorf("channel not found")
	}
	
	newPop := ef.pops[newPopID]
	if newPop == nil || !newPop.Active {
		return fmt.Errorf("invalid PoP")
	}
	
	oldPopID := channel.PopID
	channel.PopID = newPopID
	
	if oldPop := ef.pops[oldPopID]; oldPop != nil {
		oldPop.Load--
	}
	newPop.Load++
	
	return nil
}

func (ef *EdgeFabric) SelectOptimalPoP(tenantID, region string) *PoP {
	ef.mu.RLock()
	defer ef.mu.RUnlock()
	
	var bestPop *PoP
	bestScore := float64(-1)
	
	for _, pop := range ef.pops {
		if !pop.Active {
			continue
		}
		
		loadFactor := 1.0 - (float64(pop.Load) / float64(pop.Capacity))
		latencyFactor := 1.0 / (1.0 + pop.Latency.Seconds())
		regionBonus := 0.0
		if pop.Region == region {
			regionBonus = 0.5
		}
		
		score := loadFactor*0.4 + latencyFactor*0.4 + regionBonus*0.2
		
		if score > bestScore {
			bestScore = score
			bestPop = pop
		}
	}
	
	return bestPop
}

func (ef *EdgeFabric) warmupChannels(pool *ChannelPool) {
	targetWarmup := min(5, pool.MaxChannels/2)
	
	for len(pool.PreWarmed) < targetWarmup {
		channel := ef.createChannel(pool.TenantID)
		time.Sleep(pool.WarmupTime)
		channel.Ready = true
		
		ef.mu.Lock()
		pool.PreWarmed = append(pool.PreWarmed, channel)
		ef.mu.Unlock()
	}
}

func (ef *EdgeFabric) createChannel(tenantID string) *Channel {
	pop := ef.SelectOptimalPoP(tenantID, "")
	if pop == nil {
		pop = ef.getAnyAvailablePoP()
	}
	
	channel := &Channel{
		ID:        fmt.Sprintf("ch_%d", time.Now().UnixNano()),
		TenantID:  tenantID,
		PopID:     pop.ID,
		CreatedAt: time.Now(),
		Ready:     false,
	}
	
	pop.Load++
	return channel
}

func (ef *EdgeFabric) getAnyAvailablePoP() *PoP {
	for _, pop := range ef.pops {
		if pop.Active && pop.Load < pop.Capacity {
			return pop
		}
	}
	return nil
}

func NewMeshRouter() *MeshRouter {
	return &MeshRouter{
		routes: make(map[string]*Route),
	}
}

func (mr *MeshRouter) AddRoute(source, dest string, hops []string, latency time.Duration) {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	routeKey := source + "->" + dest
	mr.routes[routeKey] = &Route{
		Source:      source,
		Destination: dest,
		Hops:        hops,
		Latency:     latency,
	}
}

func (mr *MeshRouter) FindRoute(source, dest string) *Route {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	
	routeKey := source + "->" + dest
	return mr.routes[routeKey]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}