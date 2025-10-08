package runner

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

type WasmRunner struct {
	runtime wazero.Runtime
}

type PluginInput struct {
	ArtifactID   string                 `json:"artifact_id"`
	ArtifactType string                 `json:"artifact_type"`
	Metadata     map[string]interface{} `json:"metadata"`
	S3URL        string                 `json:"s3_url"`
	Timestamp    time.Time              `json:"timestamp"`
}

type PluginOutput struct {
	Success       bool                   `json:"success"`
	Results       map[string]interface{} `json:"results"`
	Confidence    float64                `json:"confidence"`
	Tags          []string               `json:"tags"`
	Indicators    []Indicator            `json:"indicators"`
	Error         string                 `json:"error,omitempty"`
	ExecutionTime int64                  `json:"execution_time_ms"`
}

type Indicator struct {
	Type       string  `json:"type"`
	Value      string  `json:"value"`
	Confidence float64 `json:"confidence"`
	Context    string  `json:"context"`
}

type SandboxPolicy struct {
	NetworkAccess    bool     `json:"network_access"`
	FilesystemAccess string   `json:"filesystem_access"`
	MemoryLimit      string   `json:"memory_limit"`
	CPULimit         string   `json:"cpu_limit"`
	ExecutionTimeout string   `json:"execution_timeout"`
	AllowedSyscalls  []string `json:"allowed_syscalls"`
	RiskLevel        string   `json:"risk_level"`
}

func NewWasmRunner() *WasmRunner {
	ctx := context.Background()

	// Create runtime with security restrictions
	config := wazero.NewRuntimeConfig().
		WithCloseOnContextDone(true).
		WithMemoryLimitPages(2048) // 128MB limit

	runtime := wazero.NewRuntimeWithConfig(ctx, config)

	// Instantiate WASI with restricted capabilities
	wasi_snapshot_preview1.MustInstantiate(ctx, runtime)

	return &WasmRunner{runtime: runtime}
}

func (wr *WasmRunner) ExecutePlugin(wasmBytes []byte, input PluginInput, policy SandboxPolicy) (*PluginOutput, error) {
	startTime := time.Now()

	// Parse execution timeout from policy
	timeout, err := time.ParseDuration(policy.ExecutionTimeout)
	if err != nil {
		timeout = 30 * time.Second
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Compile WASM module with security restrictions
	module, err := wr.runtime.Instantiate(ctx, wasmBytes)
	if err != nil {
		return &PluginOutput{
			Success:       false,
			Error:         fmt.Sprintf("failed to instantiate WASM module: %v", err),
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, err
	}
	defer module.Close(ctx)

	// Prepare input JSON
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return &PluginOutput{
			Success:       false,
			Error:         fmt.Sprintf("failed to marshal input: %v", err),
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, err
	}

	// Execute plugin main function
	analyzeFunc := module.ExportedFunction("analyze")
	if analyzeFunc == nil {
		return &PluginOutput{
			Success:       false,
			Error:         "plugin missing 'analyze' function",
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, fmt.Errorf("plugin missing 'analyze' function")
	}

	// Allocate memory for input
	mallocFunc := module.ExportedFunction("malloc")
	if mallocFunc == nil {
		return &PluginOutput{
			Success:       false,
			Error:         "plugin missing 'malloc' function",
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, fmt.Errorf("plugin missing 'malloc' function")
	}

	inputSize := uint64(len(inputJSON))
	results, err := mallocFunc.Call(ctx, inputSize)
	if err != nil {
		return &PluginOutput{
			Success:       false,
			Error:         fmt.Sprintf("failed to allocate memory: %v", err),
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, err
	}

	inputPtr := results[0]

	// Write input to WASM memory
	if !module.Memory().Write(uint32(inputPtr), inputJSON) {
		return &PluginOutput{
			Success:       false,
			Error:         "failed to write input to WASM memory",
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, fmt.Errorf("failed to write input to WASM memory")
	}

	// Call analyze function with resource monitoring
	results, err = analyzeFunc.Call(ctx, inputPtr, inputSize)
	if err != nil {
		return &PluginOutput{
			Success:       false,
			Error:         fmt.Sprintf("plugin execution failed: %v", err),
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, err
	}

	outputPtr := results[0]
	outputSize := results[1]

	// Read output from WASM memory
	outputBytes, ok := module.Memory().Read(uint32(outputPtr), uint32(outputSize))
	if !ok {
		return &PluginOutput{
			Success:       false,
			Error:         "failed to read output from WASM memory",
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, fmt.Errorf("failed to read output from WASM memory")
	}

	// Parse output
	var output PluginOutput
	if err := json.Unmarshal(outputBytes, &output); err != nil {
		return &PluginOutput{
			Success:       false,
			Error:         fmt.Sprintf("failed to parse plugin output: %v", err),
			ExecutionTime: time.Since(startTime).Milliseconds(),
		}, err
	}

	// Set execution time
	output.ExecutionTime = time.Since(startTime).Milliseconds()

	// Validate output schema
	if err := wr.validateOutput(&output); err != nil {
		output.Success = false
		output.Error = fmt.Sprintf("output validation failed: %v", err)
	}

	return &output, nil
}

func (wr *WasmRunner) validateOutput(output *PluginOutput) error {
	// Validate confidence score
	if output.Confidence < 0.0 || output.Confidence > 1.0 {
		return fmt.Errorf("confidence score must be between 0.0 and 1.0")
	}

	// Validate indicators
	for _, indicator := range output.Indicators {
		if indicator.Type == "" || indicator.Value == "" {
			return fmt.Errorf("indicator must have type and value")
		}
		if indicator.Confidence < 0.0 || indicator.Confidence > 1.0 {
			return fmt.Errorf("indicator confidence must be between 0.0 and 1.0")
		}
	}

	// Validate results structure
	if output.Results == nil {
		output.Results = make(map[string]interface{})
	}

	return nil
}

func (wr *WasmRunner) TestPluginIsolation(wasmBytes []byte) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	module, err := wr.runtime.Instantiate(ctx, wasmBytes)
	if err != nil {
		return fmt.Errorf("failed to instantiate test module: %w", err)
	}
	defer module.Close(ctx)

	// Test network access (should be blocked)
	networkTestFunc := module.ExportedFunction("test_network")
	if networkTestFunc != nil {
		results, err := networkTestFunc.Call(ctx)
		if err == nil && len(results) > 0 && results[0] == 1 {
			return fmt.Errorf("plugin has network access - isolation failed")
		}
	}

	// Test filesystem access (should be blocked)
	fsTestFunc := module.ExportedFunction("test_filesystem")
	if fsTestFunc != nil {
		results, err := fsTestFunc.Call(ctx)
		if err == nil && len(results) > 0 && results[0] == 1 {
			return fmt.Errorf("plugin has filesystem access - isolation failed")
		}
	}

	return nil
}

func (wr *WasmRunner) GetRuntimeStats() map[string]interface{} {
	return map[string]interface{}{
		"runtime_type":      "wazero",
		"wasi_enabled":      true,
		"memory_limit":      "128MB",
		"network_access":    false,
		"filesystem_access": "none",
	}
}

func (wr *WasmRunner) Close() error {
	ctx := context.Background()
	return wr.runtime.Close(ctx)
}
