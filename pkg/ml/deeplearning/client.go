// Package deeplearning provides Go client for PyTorch Deep Learning Service
package deeplearning

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client is the deep learning service client
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new deep learning service client
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ModelType represents the type of deep learning model
type ModelType string

const (
	ModelTypeAutoencoder     ModelType = "autoencoder"
	ModelTypeLSTMAutoencoder ModelType = "lstm_autoencoder"
)

// ModelConfig holds configuration for model creation
type ModelConfig struct {
	InputDim      int     `json:"input_dim"`
	LatentDim     int     `json:"latent_dim,omitempty"`
	HiddenDim     int     `json:"hidden_dim,omitempty"`
	HiddenDims    []int   `json:"hidden_dims,omitempty"`
	NumLayers     int     `json:"num_layers,omitempty"`
	Bidirectional bool    `json:"bidirectional,omitempty"`
	LearningRate  float64 `json:"learning_rate,omitempty"`
}

// TrainingParams holds training parameters
type TrainingParams struct {
	Epochs                 int     `json:"epochs,omitempty"`
	BatchSize              int     `json:"batch_size,omitempty"`
	ValidationSplit        float64 `json:"validation_split,omitempty"`
	EarlyStoppingPatience  int     `json:"early_stopping_patience,omitempty"`
}

// TrainRequest is the request for training a model
type TrainRequest struct {
	ModelType      ModelType      `json:"model_type"`
	Config         ModelConfig    `json:"config"`
	TrainingData   [][]float64    `json:"training_data"`
	TrainingParams TrainingParams `json:"training_params,omitempty"`
}

// TrainResponse is the response from training
type TrainResponse struct {
	Status      string    `json:"status"`
	ModelName   string    `json:"model_name"`
	ModelType   string    `json:"model_type"`
	ModelPath   string    `json:"model_path"`
	Threshold   *float64  `json:"threshold"`
	TrainLosses []float64 `json:"train_losses"`
	ValLosses   []float64 `json:"val_losses"`
}

// LoadRequest is the request for loading a model
type LoadRequest struct {
	ModelType string `json:"model_type"`
	ModelPath string `json:"model_path,omitempty"`
}

// LoadResponse is the response from loading a model
type LoadResponse struct {
	Status    string   `json:"status"`
	ModelName string   `json:"model_name"`
	ModelType string   `json:"model_type"`
	Threshold *float64 `json:"threshold"`
}

// PredictRequest is the request for making predictions
type PredictRequest struct {
	Data        interface{} `json:"data"` // [][]float64 or [][][]float64 for sequences
	ReturnProba bool        `json:"return_proba,omitempty"`
}

// PredictResponse is the response from predictions
type PredictResponse struct {
	Predictions         []float64 `json:"predictions"`
	ReconstructionError []float64 `json:"reconstruction_errors"`
	Threshold           *float64  `json:"threshold"`
	NumAnomalies        int       `json:"num_anomalies"`
}

// EvaluateRequest is the request for model evaluation
type EvaluateRequest struct {
	Data   [][]float64 `json:"data"`
	Labels []int       `json:"labels"`
}

// EvaluateResponse is the response from evaluation
type EvaluateResponse struct {
	Accuracy        float64           `json:"accuracy"`
	Precision       float64           `json:"precision"`
	Recall          float64           `json:"recall"`
	F1Score         float64           `json:"f1_score"`
	ROCAUC          float64           `json:"roc_auc,omitempty"`
	ConfusionMatrix map[string]int    `json:"confusion_matrix"`
}

// HealthResponse is the response from health check
type HealthResponse struct {
	Status       string   `json:"status"`
	LoadedModels []string `json:"loaded_models"`
}

// Health checks if the service is healthy
func (c *Client) Health() (*HealthResponse, error) {
	resp, err := c.httpClient.Get(c.baseURL + "/health")
	if err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("health check failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode health response: %w", err)
	}

	return &result, nil
}

// Train trains a new model
func (c *Client) Train(modelName string, req TrainRequest) (*TrainResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		fmt.Sprintf("%s/models/%s/train", c.baseURL, modelName),
		"application/json",
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, fmt.Errorf("train request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("train failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result TrainResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode train response: %w", err)
	}

	return &result, nil
}

// Load loads a trained model
func (c *Client) Load(modelName string, req LoadRequest) (*LoadResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		fmt.Sprintf("%s/models/%s/load", c.baseURL, modelName),
		"application/json",
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, fmt.Errorf("load request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("load failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result LoadResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode load response: %w", err)
	}

	return &result, nil
}

// Predict makes predictions with a loaded model
func (c *Client) Predict(modelName string, req PredictRequest) (*PredictResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		fmt.Sprintf("%s/models/%s/predict", c.baseURL, modelName),
		"application/json",
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, fmt.Errorf("predict request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("predict failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result PredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode predict response: %w", err)
	}

	return &result, nil
}

// Evaluate evaluates model performance
func (c *Client) Evaluate(modelName string, req EvaluateRequest) (*EvaluateResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		fmt.Sprintf("%s/models/%s/evaluate", c.baseURL, modelName),
		"application/json",
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, fmt.Errorf("evaluate request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("evaluate failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result EvaluateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode evaluate response: %w", err)
	}

	return &result, nil
}

// Unload unloads a model from memory
func (c *Client) Unload(modelName string) error {
	resp, err := c.httpClient.Post(
		fmt.Sprintf("%s/models/%s/unload", c.baseURL, modelName),
		"application/json",
		nil,
	)
	if err != nil {
		return fmt.Errorf("unload request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unload failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}
