package deeplearning

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	client := NewClient("http://localhost:8001")
	assert.NotNil(t, client)
	assert.Equal(t, "http://localhost:8001", client.baseURL)
	assert.NotNil(t, client.httpClient)
}

func TestHealth(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/health", r.URL.Path)
		assert.Equal(t, http.MethodGet, r.Method)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(HealthResponse{
			Status:       "healthy",
			LoadedModels: []string{"model1", "model2"},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	resp, err := client.Health()

	require.NoError(t, err)
	assert.Equal(t, "healthy", resp.Status)
	assert.Equal(t, 2, len(resp.LoadedModels))
}

func TestTrain(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/models/test_model/train", r.URL.Path)
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req TrainRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, ModelTypeAutoencoder, req.ModelType)
		assert.Equal(t, 50, req.Config.InputDim)

		threshold := 0.15
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(TrainResponse{
			Status:    "success",
			ModelName: "test_model",
			ModelType: "autoencoder",
			ModelPath: "/tmp/test_model.pt",
			Threshold: &threshold,
			TrainLosses: []float64{0.5, 0.4, 0.3},
			ValLosses:   []float64{0.45, 0.38, 0.35},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	
	// Generate test data
	testData := make([][]float64, 100)
	for i := range testData {
		testData[i] = make([]float64, 50)
		for j := range testData[i] {
			testData[i][j] = float64(i + j)
		}
	}

	req := TrainRequest{
		ModelType: ModelTypeAutoencoder,
		Config: ModelConfig{
			InputDim:  50,
			LatentDim: 16,
		},
		TrainingData: testData,
		TrainingParams: TrainingParams{
			Epochs:    10,
			BatchSize: 32,
		},
	}

	resp, err := client.Train("test_model", req)

	require.NoError(t, err)
	assert.Equal(t, "success", resp.Status)
	assert.Equal(t, "test_model", resp.ModelName)
	assert.Equal(t, "autoencoder", resp.ModelType)
	assert.NotNil(t, resp.Threshold)
	assert.Equal(t, 0.15, *resp.Threshold)
	assert.Equal(t, 3, len(resp.TrainLosses))
}

func TestLoad(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/models/test_model/load", r.URL.Path)
		assert.Equal(t, http.MethodPost, r.Method)

		var req LoadRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, "autoencoder", req.ModelType)

		threshold := 0.12
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(LoadResponse{
			Status:    "success",
			ModelName: "test_model",
			ModelType: "autoencoder",
			Threshold: &threshold,
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)

	req := LoadRequest{
		ModelType: "autoencoder",
		ModelPath: "/tmp/test_model.pt",
	}

	resp, err := client.Load("test_model", req)

	require.NoError(t, err)
	assert.Equal(t, "success", resp.Status)
	assert.Equal(t, "test_model", resp.ModelName)
	assert.NotNil(t, resp.Threshold)
}

func TestPredict(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/models/test_model/predict", r.URL.Path)
		assert.Equal(t, http.MethodPost, r.Method)

		var req PredictRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.True(t, req.ReturnProba)

		threshold := 0.12
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(PredictResponse{
			Predictions:         []float64{0.1, 0.8, 0.3},
			ReconstructionError: []float64{0.05, 0.25, 0.10},
			Threshold:           &threshold,
			NumAnomalies:        1,
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)

	testData := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	req := PredictRequest{
		Data:        testData,
		ReturnProba: true,
	}

	resp, err := client.Predict("test_model", req)

	require.NoError(t, err)
	assert.Equal(t, 3, len(resp.Predictions))
	assert.Equal(t, 3, len(resp.ReconstructionError))
	assert.Equal(t, 1, resp.NumAnomalies)
	assert.NotNil(t, resp.Threshold)
}

func TestEvaluate(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/models/test_model/evaluate", r.URL.Path)
		assert.Equal(t, http.MethodPost, r.Method)

		var req EvaluateRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, 3, len(req.Data))
		assert.Equal(t, 3, len(req.Labels))

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(EvaluateResponse{
			Accuracy:  0.95,
			Precision: 0.93,
			Recall:    0.92,
			F1Score:   0.925,
			ROCAUC:    0.94,
			ConfusionMatrix: map[string]int{
				"true_negatives":  100,
				"false_positives": 5,
				"false_negatives": 3,
				"true_positives":  92,
			},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)

	req := EvaluateRequest{
		Data: [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
		},
		Labels: []int{0, 1, 0},
	}

	resp, err := client.Evaluate("test_model", req)

	require.NoError(t, err)
	assert.Equal(t, 0.95, resp.Accuracy)
	assert.Equal(t, 0.93, resp.Precision)
	assert.Equal(t, 0.92, resp.Recall)
	assert.Equal(t, 0.925, resp.F1Score)
	assert.Equal(t, 0.94, resp.ROCAUC)
	assert.Equal(t, 100, resp.ConfusionMatrix["true_negatives"])
}

func TestUnload(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/models/test_model/unload", r.URL.Path)
		assert.Equal(t, http.MethodPost, r.Method)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "success",
			"message": "Model test_model unloaded",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	err := client.Unload("test_model")

	require.NoError(t, err)
}

func TestTrainLSTMAutoencoder(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req TrainRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, ModelTypeLSTMAutoencoder, req.ModelType)
		assert.Equal(t, 10, req.Config.InputDim)
		assert.Equal(t, 32, req.Config.HiddenDim)
		assert.True(t, req.Config.Bidirectional)

		threshold := 0.20
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(TrainResponse{
			Status:    "success",
			ModelName: "lstm_model",
			ModelType: "lstm_autoencoder",
			Threshold: &threshold,
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)

	// Generate sequential test data
	testData := make([][]float64, 50)
	for i := range testData {
		testData[i] = make([]float64, 10)
	}

	req := TrainRequest{
		ModelType: ModelTypeLSTMAutoencoder,
		Config: ModelConfig{
			InputDim:      10,
			HiddenDim:     32,
			LatentDim:     16,
			NumLayers:     2,
			Bidirectional: true,
		},
		TrainingData: testData,
		TrainingParams: TrainingParams{
			Epochs:    20,
			BatchSize: 16,
		},
	}

	resp, err := client.Train("lstm_model", req)

	require.NoError(t, err)
	assert.Equal(t, "success", resp.Status)
	assert.Equal(t, "lstm_model", resp.ModelName)
}

func TestHealthError(t *testing.T) {
	// Mock server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Internal server error"))
	}))
	defer server.Close()

	client := NewClient(server.URL)
	_, err := client.Health()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "health check failed with status 500")
}

func TestPredictModelNotFound(t *testing.T) {
	// Mock server that returns 404
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Model not found",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)

	req := PredictRequest{
		Data: [][]float64{{1.0, 2.0, 3.0}},
	}

	_, err := client.Predict("nonexistent_model", req)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "predict failed with status 404")
}
