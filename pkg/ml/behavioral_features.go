package ml

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// BehavioralPattern represents user/entity behavioral patterns
type BehavioralPattern struct {
	EntityID  string
	Timestamp time.Time

	// Access patterns
	AccessFrequency    float64 // Accesses per hour
	AccessCount        int
	UniqueResources    int
	ResourceDiversity  float64 // Entropy of resource access
	
	// Temporal patterns
	HourOfDay          int
	DayOfWeek          int
	IsBusinessHours    bool
	IsWeekend          bool
	TimeDeviation      float64 // Deviation from normal access time
	
	// Geographic patterns
	LocationCount      int
	LocationDiversity  float64
	GeoAnomaly         float64 // Distance from usual location
	
	// Activity patterns
	FailedAttempts     int
	SuccessRate        float64
	SessionDuration    float64 // Minutes
	ActionsPerSession  float64
	
	// Volume patterns
	DataVolumeRead     float64 // MB
	DataVolumeWrite    float64 // MB
	RequestRate        float64 // Requests per minute
	
	// Deviation metrics
	FrequencyDeviation float64 // From user baseline
	VolumeDeviation    float64
	PatternAnomaly     float64 // Overall pattern score
}

// BehavioralEvent represents a single behavioral event
type BehavioralEvent struct {
	EntityID    string
	Timestamp   time.Time
	Action      string
	Resource    string
	Location    string
	Success     bool
	Duration    time.Duration
	DataRead    float64
	DataWrite   float64
	Metadata    map[string]string
}

// BehavioralProfile represents the baseline profile for an entity
type BehavioralProfile struct {
	EntityID string

	// Baseline statistics
	AvgAccessFrequency float64
	StdAccessFrequency float64
	AvgSessionDuration float64
	StdSessionDuration float64
	AvgDataVolume      float64
	StdDataVolume      float64
	
	// Typical patterns
	CommonHours     []int     // Typical hours of activity
	CommonDays      []int     // Typical days of week
	CommonLocations []string  // Typical locations
	CommonResources []string  // Typical resources
	
	// Time window
	WindowStart time.Time
	WindowEnd   time.Time
	EventCount  int
	
	// Last update
	UpdatedAt time.Time
}

// BehavioralFeatureExtractor extracts behavioral features
type BehavioralFeatureExtractor struct {
	mu sync.RWMutex
	
	profiles      map[string]*BehavioralProfile // entityID -> profile
	events        map[string][]BehavioralEvent  // entityID -> events buffer
	windowSize    time.Duration
	updateInterval time.Duration
	
	// Configuration
	minEventsForProfile int
	maxEventsBuffer     int
}

// BehavioralConfig configures the extractor
type BehavioralConfig struct {
	WindowSize          time.Duration
	UpdateInterval      time.Duration
	MinEventsForProfile int
	MaxEventsBuffer     int
}

// NewBehavioralFeatureExtractor creates a new behavioral feature extractor
func NewBehavioralFeatureExtractor(config BehavioralConfig) *BehavioralFeatureExtractor {
	if config.WindowSize <= 0 {
		config.WindowSize = 24 * time.Hour
	}
	if config.UpdateInterval <= 0 {
		config.UpdateInterval = 1 * time.Hour
	}
	if config.MinEventsForProfile <= 0 {
		config.MinEventsForProfile = 10
	}
	if config.MaxEventsBuffer <= 0 {
		config.MaxEventsBuffer = 1000
	}

	return &BehavioralFeatureExtractor{
		profiles:            make(map[string]*BehavioralProfile),
		events:              make(map[string][]BehavioralEvent),
		windowSize:          config.WindowSize,
		updateInterval:      config.UpdateInterval,
		minEventsForProfile: config.MinEventsForProfile,
		maxEventsBuffer:     config.MaxEventsBuffer,
	}
}

// AddEvent adds a behavioral event
func (bfe *BehavioralFeatureExtractor) AddEvent(event BehavioralEvent) {
	bfe.mu.Lock()
	defer bfe.mu.Unlock()

	if _, exists := bfe.events[event.EntityID]; !exists {
		bfe.events[event.EntityID] = make([]BehavioralEvent, 0)
	}

	bfe.events[event.EntityID] = append(bfe.events[event.EntityID], event)

	// Trim buffer if needed
	if len(bfe.events[event.EntityID]) > bfe.maxEventsBuffer {
		bfe.events[event.EntityID] = bfe.events[event.EntityID][len(bfe.events[event.EntityID])-bfe.maxEventsBuffer:]
	}

	// Clean old events outside window
	cutoff := time.Now().Add(-bfe.windowSize)
	validEvents := make([]BehavioralEvent, 0)
	for _, e := range bfe.events[event.EntityID] {
		if e.Timestamp.After(cutoff) {
			validEvents = append(validEvents, e)
		}
	}
	bfe.events[event.EntityID] = validEvents
}

// ExtractFeatures extracts behavioral features for an entity
func (bfe *BehavioralFeatureExtractor) ExtractFeatures(entityID string) (*BehavioralPattern, error) {
	bfe.mu.RLock()
	defer bfe.mu.RUnlock()

	events, exists := bfe.events[entityID]
	if !exists || len(events) == 0 {
		return nil, fmt.Errorf("no events found for entity: %s", entityID)
	}

	pattern := &BehavioralPattern{
		EntityID:  entityID,
		Timestamp: time.Now(),
	}

	// Extract access patterns
	pattern.AccessCount = len(events)
	pattern.UniqueResources = bfe.countUniqueResources(events)
	pattern.ResourceDiversity = bfe.calculateResourceDiversity(events)

	// Calculate access frequency (events per hour)
	if len(events) > 1 {
		duration := events[len(events)-1].Timestamp.Sub(events[0].Timestamp).Hours()
		if duration > 0 {
			pattern.AccessFrequency = float64(len(events)) / duration
		}
	}

	// Extract temporal patterns
	if len(events) > 0 {
		lastEvent := events[len(events)-1]
		pattern.HourOfDay = lastEvent.Timestamp.Hour()
		pattern.DayOfWeek = int(lastEvent.Timestamp.Weekday())
		pattern.IsBusinessHours = bfe.isBusinessHours(lastEvent.Timestamp)
		pattern.IsWeekend = bfe.isWeekend(lastEvent.Timestamp)
	}

	// Extract location patterns
	pattern.LocationCount = bfe.countUniqueLocations(events)
	pattern.LocationDiversity = bfe.calculateLocationDiversity(events)

	// Extract activity patterns
	pattern.FailedAttempts = bfe.countFailedAttempts(events)
	if pattern.AccessCount > 0 {
		pattern.SuccessRate = float64(pattern.AccessCount-pattern.FailedAttempts) / float64(pattern.AccessCount)
	}

	// Calculate session metrics
	pattern.SessionDuration = bfe.calculateAvgSessionDuration(events)
	pattern.ActionsPerSession = bfe.calculateActionsPerSession(events)

	// Calculate volume metrics
	pattern.DataVolumeRead = bfe.calculateTotalDataRead(events)
	pattern.DataVolumeWrite = bfe.calculateTotalDataWrite(events)
	pattern.RequestRate = bfe.calculateRequestRate(events)

	// Calculate deviations from baseline if profile exists
	if profile, hasProfile := bfe.profiles[entityID]; hasProfile {
		pattern.FrequencyDeviation = bfe.calculateDeviation(pattern.AccessFrequency, 
			profile.AvgAccessFrequency, profile.StdAccessFrequency)
		
		totalVolume := pattern.DataVolumeRead + pattern.DataVolumeWrite
		pattern.VolumeDeviation = bfe.calculateDeviation(totalVolume, 
			profile.AvgDataVolume, profile.StdDataVolume)
		
		pattern.TimeDeviation = bfe.calculateTimeDeviation(events, profile)
		pattern.GeoAnomaly = bfe.calculateGeoAnomaly(events, profile)
		
		// Overall pattern anomaly score (weighted combination)
		pattern.PatternAnomaly = 0.3*pattern.FrequencyDeviation + 
			0.3*pattern.VolumeDeviation + 
			0.2*pattern.TimeDeviation + 
			0.2*pattern.GeoAnomaly
	}

	return pattern, nil
}

// UpdateProfile updates the baseline profile for an entity
func (bfe *BehavioralFeatureExtractor) UpdateProfile(entityID string) error {
	bfe.mu.Lock()
	defer bfe.mu.Unlock()

	events, exists := bfe.events[entityID]
	if !exists || len(events) < bfe.minEventsForProfile {
		return fmt.Errorf("insufficient events for profile: need %d, have %d", 
			bfe.minEventsForProfile, len(events))
	}

	profile := &BehavioralProfile{
		EntityID:   entityID,
		EventCount: len(events),
		UpdatedAt:  time.Now(),
	}

	if len(events) > 0 {
		profile.WindowStart = events[0].Timestamp
		profile.WindowEnd = events[len(events)-1].Timestamp
	}

	// Calculate frequency statistics
	frequencies := bfe.calculateHourlyFrequencies(events)
	profile.AvgAccessFrequency = bfe.calculateMeanFloat(frequencies)
	profile.StdAccessFrequency = bfe.calculateStdDevFloat(frequencies, profile.AvgAccessFrequency)

	// Calculate session duration statistics
	durations := bfe.extractSessionDurations(events)
	profile.AvgSessionDuration = bfe.calculateMeanFloat(durations)
	profile.StdSessionDuration = bfe.calculateStdDevFloat(durations, profile.AvgSessionDuration)

	// Calculate volume statistics
	volumes := bfe.extractDataVolumes(events)
	profile.AvgDataVolume = bfe.calculateMeanFloat(volumes)
	profile.StdDataVolume = bfe.calculateStdDevFloat(volumes, profile.AvgDataVolume)

	// Extract typical patterns
	profile.CommonHours = bfe.extractCommonHours(events)
	profile.CommonDays = bfe.extractCommonDays(events)
	profile.CommonLocations = bfe.extractCommonLocations(events)
	profile.CommonResources = bfe.extractCommonResources(events)

	bfe.profiles[entityID] = profile
	return nil
}

// GetProfile retrieves the profile for an entity
func (bfe *BehavioralFeatureExtractor) GetProfile(entityID string) (*BehavioralProfile, error) {
	bfe.mu.RLock()
	defer bfe.mu.RUnlock()

	profile, exists := bfe.profiles[entityID]
	if !exists {
		return nil, fmt.Errorf("no profile found for entity: %s", entityID)
	}

	return profile, nil
}

// ToVector converts behavioral pattern to feature vector
func (bp *BehavioralPattern) ToVector() []float64 {
	vector := make([]float64, 20)
	
	vector[0] = bp.AccessFrequency
	vector[1] = float64(bp.AccessCount)
	vector[2] = float64(bp.UniqueResources)
	vector[3] = bp.ResourceDiversity
	vector[4] = float64(bp.HourOfDay)
	vector[5] = float64(bp.DayOfWeek)
	vector[6] = boolToFloat(bp.IsBusinessHours)
	vector[7] = boolToFloat(bp.IsWeekend)
	vector[8] = bp.TimeDeviation
	vector[9] = float64(bp.LocationCount)
	vector[10] = bp.LocationDiversity
	vector[11] = bp.GeoAnomaly
	vector[12] = float64(bp.FailedAttempts)
	vector[13] = bp.SuccessRate
	vector[14] = bp.SessionDuration
	vector[15] = bp.ActionsPerSession
	vector[16] = bp.DataVolumeRead
	vector[17] = bp.DataVolumeWrite
	vector[18] = bp.RequestRate
	vector[19] = bp.PatternAnomaly
	
	return vector
}

// Helper methods

func (bfe *BehavioralFeatureExtractor) countUniqueResources(events []BehavioralEvent) int {
	resources := make(map[string]bool)
	for _, e := range events {
		resources[e.Resource] = true
	}
	return len(resources)
}

func (bfe *BehavioralFeatureExtractor) calculateResourceDiversity(events []BehavioralEvent) float64 {
	if len(events) == 0 {
		return 0
	}
	
	counts := make(map[string]int)
	for _, e := range events {
		counts[e.Resource]++
	}
	
	// Calculate Shannon entropy
	entropy := 0.0
	total := float64(len(events))
	for _, count := range counts {
		p := float64(count) / total
		entropy -= p * math.Log2(p)
	}
	
	return entropy
}

func (bfe *BehavioralFeatureExtractor) isBusinessHours(t time.Time) bool {
	hour := t.Hour()
	return hour >= 9 && hour < 17
}

func (bfe *BehavioralFeatureExtractor) isWeekend(t time.Time) bool {
	day := t.Weekday()
	return day == time.Saturday || day == time.Sunday
}

func (bfe *BehavioralFeatureExtractor) countUniqueLocations(events []BehavioralEvent) int {
	locations := make(map[string]bool)
	for _, e := range events {
		if e.Location != "" {
			locations[e.Location] = true
		}
	}
	return len(locations)
}

func (bfe *BehavioralFeatureExtractor) calculateLocationDiversity(events []BehavioralEvent) float64 {
	counts := make(map[string]int)
	for _, e := range events {
		if e.Location != "" {
			counts[e.Location]++
		}
	}
	
	if len(counts) == 0 {
		return 0
	}
	
	entropy := 0.0
	total := 0
	for _, count := range counts {
		total += count
	}
	
	for _, count := range counts {
		p := float64(count) / float64(total)
		entropy -= p * math.Log2(p)
	}
	
	return entropy
}

func (bfe *BehavioralFeatureExtractor) countFailedAttempts(events []BehavioralEvent) int {
	count := 0
	for _, e := range events {
		if !e.Success {
			count++
		}
	}
	return count
}

func (bfe *BehavioralFeatureExtractor) calculateAvgSessionDuration(events []BehavioralEvent) float64 {
	if len(events) == 0 {
		return 0
	}
	
	totalDuration := time.Duration(0)
	for _, e := range events {
		totalDuration += e.Duration
	}
	
	return totalDuration.Minutes() / float64(len(events))
}

func (bfe *BehavioralFeatureExtractor) calculateActionsPerSession(events []BehavioralEvent) float64 {
	// Simplified: assume each event is an action
	return float64(len(events))
}

func (bfe *BehavioralFeatureExtractor) calculateTotalDataRead(events []BehavioralEvent) float64 {
	total := 0.0
	for _, e := range events {
		total += e.DataRead
	}
	return total
}

func (bfe *BehavioralFeatureExtractor) calculateTotalDataWrite(events []BehavioralEvent) float64 {
	total := 0.0
	for _, e := range events {
		total += e.DataWrite
	}
	return total
}

func (bfe *BehavioralFeatureExtractor) calculateRequestRate(events []BehavioralEvent) float64 {
	if len(events) < 2 {
		return 0
	}
	
	duration := events[len(events)-1].Timestamp.Sub(events[0].Timestamp).Minutes()
	if duration <= 0 {
		return 0
	}
	
	return float64(len(events)) / duration
}

func (bfe *BehavioralFeatureExtractor) calculateDeviation(value, mean, std float64) float64 {
	if std == 0 {
		return 0
	}
	return math.Abs(value-mean) / std
}

func (bfe *BehavioralFeatureExtractor) calculateTimeDeviation(events []BehavioralEvent, profile *BehavioralProfile) float64 {
	if len(events) == 0 {
		return 0
	}
	
	// Check if current hour is in common hours
	lastEvent := events[len(events)-1]
	hour := lastEvent.Timestamp.Hour()
	
	for _, commonHour := range profile.CommonHours {
		if hour == commonHour {
			return 0 // Normal time
		}
	}
	
	return 1.0 // Unusual time
}

func (bfe *BehavioralFeatureExtractor) calculateGeoAnomaly(events []BehavioralEvent, profile *BehavioralProfile) float64 {
	if len(events) == 0 {
		return 0
	}
	
	// Check if current location is in common locations
	for _, e := range events {
		if e.Location == "" {
			continue
		}
		
		for _, commonLoc := range profile.CommonLocations {
			if e.Location == commonLoc {
				return 0 // Normal location
			}
		}
	}
	
	return 1.0 // Unusual location
}

func (bfe *BehavioralFeatureExtractor) calculateHourlyFrequencies(events []BehavioralEvent) []float64 {
	if len(events) < 2 {
		return []float64{0}
	}
	
	// Group by hour and count
	hourCounts := make(map[int]int)
	for _, e := range events {
		hour := e.Timestamp.Hour()
		hourCounts[hour]++
	}
	
	frequencies := make([]float64, 0)
	for _, count := range hourCounts {
		frequencies = append(frequencies, float64(count))
	}
	
	return frequencies
}

func (bfe *BehavioralFeatureExtractor) extractSessionDurations(events []BehavioralEvent) []float64 {
	durations := make([]float64, 0)
	for _, e := range events {
		durations = append(durations, e.Duration.Minutes())
	}
	return durations
}

func (bfe *BehavioralFeatureExtractor) extractDataVolumes(events []BehavioralEvent) []float64 {
	volumes := make([]float64, 0)
	for _, e := range events {
		volumes = append(volumes, e.DataRead+e.DataWrite)
	}
	return volumes
}

func (bfe *BehavioralFeatureExtractor) extractCommonHours(events []BehavioralEvent) []int {
	hourCounts := make(map[int]int)
	for _, e := range events {
		hourCounts[e.Timestamp.Hour()]++
	}
	
	// Get top 3 hours
	type hourCount struct {
		hour  int
		count int
	}
	hc := make([]hourCount, 0)
	for h, c := range hourCounts {
		hc = append(hc, hourCount{h, c})
	}
	
	// Sort by count (simple bubble sort for small array)
	for i := 0; i < len(hc); i++ {
		for j := i + 1; j < len(hc); j++ {
			if hc[j].count > hc[i].count {
				hc[i], hc[j] = hc[j], hc[i]
			}
		}
	}
	
	result := make([]int, 0)
	for i := 0; i < len(hc) && i < 3; i++ {
		result = append(result, hc[i].hour)
	}
	
	return result
}

func (bfe *BehavioralFeatureExtractor) extractCommonDays(events []BehavioralEvent) []int {
	dayCounts := make(map[int]int)
	for _, e := range events {
		dayCounts[int(e.Timestamp.Weekday())]++
	}
	
	result := make([]int, 0)
	for day := range dayCounts {
		result = append(result, day)
	}
	
	return result
}

func (bfe *BehavioralFeatureExtractor) extractCommonLocations(events []BehavioralEvent) []string {
	locationCounts := make(map[string]int)
	for _, e := range events {
		if e.Location != "" {
			locationCounts[e.Location]++
		}
	}
	
	// Get top 3 locations
	type locCount struct {
		loc   string
		count int
	}
	lc := make([]locCount, 0)
	for l, c := range locationCounts {
		lc = append(lc, locCount{l, c})
	}
	
	for i := 0; i < len(lc); i++ {
		for j := i + 1; j < len(lc); j++ {
			if lc[j].count > lc[i].count {
				lc[i], lc[j] = lc[j], lc[i]
			}
		}
	}
	
	result := make([]string, 0)
	for i := 0; i < len(lc) && i < 3; i++ {
		result = append(result, lc[i].loc)
	}
	
	return result
}

func (bfe *BehavioralFeatureExtractor) extractCommonResources(events []BehavioralEvent) []string {
	resourceCounts := make(map[string]int)
	for _, e := range events {
		if e.Resource != "" {
			resourceCounts[e.Resource]++
		}
	}
	
	result := make([]string, 0)
	for resource := range resourceCounts {
		result = append(result, resource)
	}
	
	return result
}

func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

func (bfe *BehavioralFeatureExtractor) calculateMeanFloat(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (bfe *BehavioralFeatureExtractor) calculateStdDevFloat(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sumSq := 0.0
	for _, v := range values {
		diff := v - mean
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(len(values)))
}
