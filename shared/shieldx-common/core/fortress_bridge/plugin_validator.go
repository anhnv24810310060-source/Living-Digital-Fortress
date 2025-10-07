package fortress_bridge

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"time"

	_ "github.com/lib/pq"
)

type PluginValidator struct {
	db *sql.DB
}

type Plugin struct {
	ID            string    `json:"id" db:"id"`
	Owner         string    `json:"owner" db:"owner"`
	Version       string    `json:"version" db:"version"`
	WasmHash      string    `json:"wasm_hash" db:"wasm_hash"`
	CosignSig     string    `json:"cosign_sig" db:"cosign_sig"`
	SBOM          string    `json:"sbom" db:"sbom"`
	Verified      bool      `json:"verified" db:"verified"`
	Status        string    `json:"status" db:"status"`
	CreatedAt     time.Time `json:"created_at" db:"created_at"`
	TrivyScan     string    `json:"trivy_scan" db:"trivy_scan"`
	SandboxPolicy string    `json:"sandbox_policy" db:"sandbox_policy"`
}

type ValidationResult struct {
	Valid       bool     `json:"valid"`
	Errors      []string `json:"errors"`
	CosignValid bool     `json:"cosign_valid"`
	SBOMValid   bool     `json:"sbom_valid"`
	TrivyClean  bool     `json:"trivy_clean"`
	RiskScore   float64  `json:"risk_score"`
}

func NewPluginValidator(dbURL string) (*PluginValidator, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	validator := &PluginValidator{db: db}
	if err := validator.migrate(); err != nil {
		return nil, fmt.Errorf("migration failed: %w", err)
	}

	return validator, nil
}

func (pv *PluginValidator) migrate() error {
	query := `
	CREATE TABLE IF NOT EXISTS plugins (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		owner VARCHAR(255) NOT NULL,
		version VARCHAR(50) NOT NULL,
		wasm_hash VARCHAR(64) NOT NULL,
		cosign_sig TEXT NOT NULL,
		sbom TEXT NOT NULL,
		verified BOOLEAN DEFAULT FALSE,
		status VARCHAR(50) DEFAULT 'pending',
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		trivy_scan TEXT,
		sandbox_policy TEXT
	);

	CREATE INDEX IF NOT EXISTS idx_plugins_owner ON plugins(owner);
	CREATE INDEX IF NOT EXISTS idx_plugins_verified ON plugins(verified);
	CREATE INDEX IF NOT EXISTS idx_plugins_status ON plugins(status);
	CREATE UNIQUE INDEX IF NOT EXISTS idx_plugins_hash ON plugins(wasm_hash);`

	_, err := pv.db.Exec(query)
	return err
}

func (pv *PluginValidator) ValidatePlugin(wasmData []byte, cosignSig, sbom, owner, version string) (*ValidationResult, error) {
	result := &ValidationResult{
		Valid:  false,
		Errors: []string{},
	}

	wasmHash := sha256.Sum256(wasmData)
	wasmHashStr := hex.EncodeToString(wasmHash[:])

	// Verify Cosign signature
	cosignValid, err := pv.verifyCosignSignature(wasmData, cosignSig)
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("Cosign verification failed: %v", err))
	}
	result.CosignValid = cosignValid

	// Validate SBOM
	sbomValid, err := pv.validateSBOM(sbom)
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("SBOM validation failed: %v", err))
	}
	result.SBOMValid = sbomValid

	// Run Trivy scan
	trivyClean, trivyReport, err := pv.runTrivyScan(wasmData)
	if err != nil {
		log.Printf("Trivy scan warning: %v", err)
	}
	result.TrivyClean = trivyClean

	// Calculate risk score
	result.RiskScore = pv.calculateRiskScore(result)

	// Overall validation
	result.Valid = result.CosignValid && result.SBOMValid && result.TrivyClean && result.RiskScore < 0.5

	// Store plugin in database
	plugin := &Plugin{
		Owner:         owner,
		Version:       version,
		WasmHash:      wasmHashStr,
		CosignSig:     cosignSig,
		SBOM:          sbom,
		Verified:      result.Valid,
		Status:        getStatusFromResult(result),
		TrivyScan:     trivyReport,
		SandboxPolicy: pv.generateSandboxPolicy(result),
	}

	if err := pv.storePlugin(plugin); err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("Failed to store plugin: %v", err))
	}

	return result, nil
}

func (pv *PluginValidator) verifyCosignSignature(wasmData []byte, signature string) (bool, error) {
	// Create temp files for cosign verification
	tmpWasm := fmt.Sprintf("/tmp/plugin_%d.wasm", time.Now().UnixNano())
	tmpSig := fmt.Sprintf("/tmp/plugin_%d.sig", time.Now().UnixNano())

	// Use cosign to verify signature
	cmd := exec.Command("cosign", "verify-blob", 
		"--signature", tmpSig,
		"--certificate-identity-regexp", ".*",
		"--certificate-oidc-issuer-regexp", ".*",
		tmpWasm)

	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Cosign verification failed: %s", string(output))
		return false, err
	}

	log.Printf("Cosign verification successful")
	return true, nil
}

func (pv *PluginValidator) validateSBOM(sbom string) (bool, error) {
	var sbomData map[string]interface{}
	if err := json.Unmarshal([]byte(sbom), &sbomData); err != nil {
		return false, fmt.Errorf("invalid SBOM JSON: %w", err)
	}

	// Check required SBOM fields
	requiredFields := []string{"bomFormat", "specVersion", "components"}
	for _, field := range requiredFields {
		if _, exists := sbomData[field]; !exists {
			return false, fmt.Errorf("missing required SBOM field: %s", field)
		}
	}

	// Validate components
	components, ok := sbomData["components"].([]interface{})
	if !ok {
		return false, fmt.Errorf("invalid components format in SBOM")
	}

	if len(components) == 0 {
		return false, fmt.Errorf("SBOM must contain at least one component")
	}

	// Check for blacklisted components
	for _, comp := range components {
		component, ok := comp.(map[string]interface{})
		if !ok {
			continue
		}

		if name, exists := component["name"]; exists {
			if pv.isBlacklistedComponent(name.(string)) {
				return false, fmt.Errorf("blacklisted component detected: %s", name)
			}
		}
	}

	return true, nil
}

func (pv *PluginValidator) runTrivyScan(wasmData []byte) (bool, string, error) {
	tmpFile := fmt.Sprintf("/tmp/plugin_%d.wasm", time.Now().UnixNano())
	
	// Run Trivy scan
	cmd := exec.Command("trivy", "fs", "--format", "json", tmpFile)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return false, string(output), err
	}

	// Parse Trivy results
	var trivyResult map[string]interface{}
	if err := json.Unmarshal(output, &trivyResult); err != nil {
		return false, string(output), err
	}

	// Check for high/critical vulnerabilities
	results, ok := trivyResult["Results"].([]interface{})
	if !ok {
		return true, string(output), nil
	}

	for _, result := range results {
		r, ok := result.(map[string]interface{})
		if !ok {
			continue
		}

		vulnerabilities, exists := r["Vulnerabilities"]
		if !exists {
			continue
		}

		vulns, ok := vulnerabilities.([]interface{})
		if !ok {
			continue
		}

		for _, vuln := range vulns {
			v, ok := vuln.(map[string]interface{})
			if !ok {
				continue
			}

			severity, exists := v["Severity"]
			if !exists {
				continue
			}

			if severity == "HIGH" || severity == "CRITICAL" {
				return false, string(output), nil
			}
		}
	}

	return true, string(output), nil
}

func (pv *PluginValidator) calculateRiskScore(result *ValidationResult) float64 {
	score := 0.0

	if !result.CosignValid {
		score += 0.4
	}

	if !result.SBOMValid {
		score += 0.3
	}

	if !result.TrivyClean {
		score += 0.3
	}

	return score
}

func (pv *PluginValidator) generateSandboxPolicy(result *ValidationResult) string {
	policy := map[string]interface{}{
		"network_access":     false,
		"filesystem_access":  "none",
		"memory_limit":       "128MB",
		"cpu_limit":          "100m",
		"execution_timeout":  "30s",
		"allowed_syscalls":   []string{"read", "write", "exit"},
		"risk_level":         getRiskLevel(result.RiskScore),
	}

	policyJSON, _ := json.Marshal(policy)
	return string(policyJSON)
}

func (pv *PluginValidator) storePlugin(plugin *Plugin) error {
	query := `
	INSERT INTO plugins (owner, version, wasm_hash, cosign_sig, sbom, verified, status, trivy_scan, sandbox_policy)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	RETURNING id, created_at`

	err := pv.db.QueryRow(query, plugin.Owner, plugin.Version, plugin.WasmHash, 
		plugin.CosignSig, plugin.SBOM, plugin.Verified, plugin.Status, 
		plugin.TrivyScan, plugin.SandboxPolicy).Scan(&plugin.ID, &plugin.CreatedAt)

	return err
}

func (pv *PluginValidator) GetPlugin(id string) (*Plugin, error) {
	query := `
	SELECT id, owner, version, wasm_hash, cosign_sig, sbom, verified, status, 
		   created_at, trivy_scan, sandbox_policy
	FROM plugins WHERE id = $1`

	plugin := &Plugin{}
	err := pv.db.QueryRow(query, id).Scan(
		&plugin.ID, &plugin.Owner, &plugin.Version, &plugin.WasmHash,
		&plugin.CosignSig, &plugin.SBOM, &plugin.Verified, &plugin.Status,
		&plugin.CreatedAt, &plugin.TrivyScan, &plugin.SandboxPolicy,
	)

	if err == sql.ErrNoRows {
		return nil, nil
	}

	return plugin, err
}

func (pv *PluginValidator) isBlacklistedComponent(name string) bool {
	blacklist := []string{
		"malicious-lib",
		"crypto-miner",
		"backdoor-util",
		"keylogger",
		"rootkit",
	}

	for _, blocked := range blacklist {
		if name == blocked {
			return true
		}
	}
	return false
}

func getStatusFromResult(result *ValidationResult) string {
	if result.Valid {
		return "verified"
	}
	if len(result.Errors) > 0 {
		return "rejected"
	}
	return "pending"
}

func getRiskLevel(score float64) string {
	if score < 0.3 {
		return "low"
	} else if score < 0.7 {
		return "medium"
	}
	return "high"
}

func (pv *PluginValidator) Close() error {
	return pv.db.Close()
}