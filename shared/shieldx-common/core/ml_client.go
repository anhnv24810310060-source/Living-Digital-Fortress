package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

type MLClient struct {
	featureStoreURL string
	httpClient      *http.Client
	metrics         *MLMetrics
}

type MLMetrics struct {
	RequestsTotal   int64 `json:"requests_total"`
	RequestsSuccess int64 `json:"requests_success"`
	RequestsFailed  int64 `json:"requests_failed"`
	AvgLatencyMs    float64 `json:"avg_latency_ms"`
}

type PluginOutputML struct {
	PluginID      string                 `json:"plugin_id"`
	ArtifactID    string                 `json:"artifact_id"`
	Success       bool                   `json:"success"`
	Results       map[string]interface{} `json:"results"`
	Confidence    float64                `json:"confidence"`
	Tags          []string               `json:"tags"`
	Indicators    []IndicatorML          `json:"indicators"`
	ExecutionTime int64                  `json:"execution_time"`
	Timestamp     string                 `json:"timestamp"`
}

type IndicatorML struct {
	Type       string  `json:"type"`
	Value      string  `json:"value"`
	Confidence float64 `json:"confidence"`
	Context    string  `json:"context"`
}

type MLResponse struct {
	Success      bool   `json:"success"`
	Message      string `json:"message,omitempty"`
	ProcessID    string `json:"process_id,omitempty"`
	FeatureCount int    `json:"feature_count,omitempty"`
	Error        string `json:"error,omitempty"`
}

type TrainingDataResponse struct {
	Success       bool        `json:"success"`
	FeatureMatrix [][]float64 `json:"feature_matrix"`
	Labels        []float64   `json:"labels"`
	FeatureNames  []string    `json:"feature_names"`
	Count         int         `json:"count"`
	Error         string      `json:"error,omitempty"`
}

func NewMLClient(featureStoreURL string) *MLClient {
	return &MLClient{
		featureStoreURL: featureStoreURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		metrics: &MLMetrics{},
	}
}

func (ml *MLClient) SendPluginOutput(pluginOutput PluginOutputML) error {
	startTime := time.Now()
	ml.metrics.RequestsTotal++

	// Ensure timestamp is set
	if pluginOutput.Timestamp == "" {
		pluginOutput.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}

	// Validate plugin output
	if err := ml.validatePluginOutput(pluginOutput); err != nil {
		ml.metrics.RequestsFailed++
		return fmt.Errorf("validation failed: %w", err)
	}

	// Convert to JSON
	jsonData, err := json.Marshal(pluginOutput)
	if err != nil {
		ml.metrics.RequestsFailed++
		return fmt.Errorf("failed to marshal plugin output: %w", err)
	}

	// Send to feature store
	resp, err := ml.httpClient.Post(
		ml.featureStoreURL+"/process",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		ml.metrics.RequestsFailed++
		return fmt.Errorf("failed to send to feature store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		ml.metrics.RequestsFailed++
		return fmt.Errorf("feature store returned status %d", resp.StatusCode)
	}

	var mlResp MLResponse
	if err := json.NewDecoder(resp.Body).Decode(&mlResp); err != nil {
		ml.metrics.RequestsFailed++
		return fmt.Errorf("failed to decode ML response: %w", err)
	}

	if !mlResp.Success {
		ml.metrics.RequestsFailed++
		return fmt.Errorf("ML processing failed: %s", mlResp.Error)
	}

	// Update metrics
	ml.metrics.RequestsSuccess++
	latency := time.Since(startTime).Milliseconds()
	ml.updateAverageLatency(float64(latency))

	log.Printf("Plugin output sent to ML pipeline: %s (features: %d)", 
		mlResp.ProcessID, mlResp.FeatureCount)
	return nil
}

func (ml *MLClient) SendBatchOutputs(outputs []PluginOutputML) error {
	successCount := 0
	for i, output := range outputs {
		if err := ml.SendPluginOutput(output); err != nil {
			log.Printf("Failed to send plugin output %d (%s): %v", i, output.ArtifactID, err)
		} else {
			successCount++
		}
	}

	log.Printf("Batch processing completed: %d/%d successful", successCount, len(outputs))
	return nil
}

func (ml *MLClient) GetTrainingData(limit int) (*TrainingDataResponse, error) {
	url := fmt.Sprintf("%s/training-data?limit=%d", ml.featureStoreURL, limit)
	
	resp, err := ml.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get training data: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("feature store returned status %d", resp.StatusCode)
	}

	var trainingResp TrainingDataResponse
	if err := json.NewDecoder(resp.Body).Decode(&trainingResp); err != nil {
		return nil, fmt.Errorf("failed to decode training data response: %w", err)
	}

	if !trainingResp.Success {
		return nil, fmt.Errorf("training data request failed: %s", trainingResp.Error)
	}

	return &trainingResp, nil
}

func (ml *MLClient) GetHealthStatus() (map[string]interface{}, error) {
	resp, err := ml.httpClient.Get(ml.featureStoreURL + "/health")
	if err != nil {
		return nil, fmt.Errorf("failed to get health status: %w", err)
	}
	defer resp.Body.Close()

	var health map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil, fmt.Errorf("failed to decode health response: %w", err)
	}

	return health, nil
}

func (ml *MLClient) GetMetrics() *MLMetrics {
	return &MLMetrics{
		RequestsTotal:   ml.metrics.RequestsTotal,
		RequestsSuccess: ml.metrics.RequestsSuccess,
		RequestsFailed:  ml.metrics.RequestsFailed,
		AvgLatencyMs:    ml.metrics.AvgLatencyMs,
	}
}

func (ml *MLClient) validatePluginOutput(output PluginOutputML) error {
	if output.PluginID == "" {
		return fmt.Errorf("plugin_id is required")
	}

	if output.ArtifactID == "" {
		return fmt.Errorf("artifact_id is required")
	}

	if output.Confidence < 0.0 || output.Confidence > 1.0 {
		return fmt.Errorf("confidence must be between 0.0 and 1.0")
	}

	if output.ExecutionTime < 0 {
		return fmt.Errorf("execution_time must be non-negative")
	}

	// Validate indicators
	for i, indicator := range output.Indicators {
		if indicator.Type == "" {
			return fmt.Errorf("indicator %d: type is required", i)
		}
		if indicator.Value == "" {
			return fmt.Errorf("indicator %d: value is required", i)
		}
		if indicator.Confidence < 0.0 || indicator.Confidence > 1.0 {
			return fmt.Errorf("indicator %d: confidence must be between 0.0 and 1.0", i)
		}
	}

	return nil
}

func (ml *MLClient) updateAverageLatency(latency float64) {
	// Simple exponential moving average
	alpha := 0.1
	ml.metrics.AvgLatencyMs = (1-alpha)*ml.metrics.AvgLatencyMs + alpha*latency
}

// Helper function to convert from sandbox runner output
func ConvertToMLOutput(pluginID, artifactID string, output interface{}) (PluginOutputML, error) {
	// This would be implemented based on the actual output structure
	// from the WASM runner
	
	mlOutput := PluginOutputML{
		PluginID:   pluginID,
		ArtifactID: artifactID,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
	}

	// Convert based on output type
	switch v := output.(type) {
	case map[string]interface{}:
		if success, ok := v["success"].(bool); ok {
			mlOutput.Success = success
		}
		if confidence, ok := v["confidence"].(float64); ok {
			mlOutput.Confidence = confidence
		}
		if results, ok := v["results"].(map[string]interface{}); ok {
			mlOutput.Results = results
		}
		if tags, ok := v["tags"].([]interface{}); ok {
			for _, tag := range tags {
				if tagStr, ok := tag.(string); ok {
					mlOutput.Tags = append(mlOutput.Tags, tagStr)
				}
			}
		}
		if indicators, ok := v["indicators"].([]interface{}); ok {
			for _, ind := range indicators {
				if indMap, ok := ind.(map[string]interface{}); ok {
					indicator := IndicatorML{}
					if iType, ok := indMap["type"].(string); ok {
						indicator.Type = iType
					}
					if value, ok := indMap["value"].(string); ok {
						indicator.Value = value
					}
					if conf, ok := indMap["confidence"].(float64); ok {
						indicator.Confidence = conf
					}
					if context, ok := indMap["context"].(string); ok {
						indicator.Context = context
					}
					mlOutput.Indicators = append(mlOutput.Indicators, indicator)
				}
			}
		}
		if execTime, ok := v["execution_time"].(float64); ok {
			mlOutput.ExecutionTime = int64(execTime)
		}
	default:
		return mlOutput, fmt.Errorf("unsupported output type: %T", output)
	}

	return mlOutput, nil
}