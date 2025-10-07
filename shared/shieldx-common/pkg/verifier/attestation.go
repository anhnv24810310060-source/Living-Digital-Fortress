package verifier

import (
	"crypto/sha256"
	"encoding/json"
	"time"
)

type SLSAAttestation struct {
	PredicateType string      `json:"predicateType"`
	Subject       []Subject   `json:"subject"`
	Predicate     Predicate   `json:"predicate"`
}

type Subject struct {
	Name   string            `json:"name"`
	Digest map[string]string `json:"digest"`
}

type Predicate struct {
	Builder     Builder     `json:"builder"`
	BuildType   string      `json:"buildType"`
	Invocation  Invocation  `json:"invocation"`
	BuildConfig BuildConfig `json:"buildConfig"`
	Materials   []Material  `json:"materials"`
}

type Builder struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

type Invocation struct {
	ConfigSource ConfigSource `json:"configSource"`
	Parameters   interface{}  `json:"parameters"`
}

type ConfigSource struct {
	URI    string            `json:"uri"`
	Digest map[string]string `json:"digest"`
}

type BuildConfig struct {
	Steps []BuildStep `json:"steps"`
}

type BuildStep struct {
	Command []string          `json:"command"`
	Env     map[string]string `json:"env"`
}

type Material struct {
	URI    string            `json:"uri"`
	Digest map[string]string `json:"digest"`
}

type AttestationGenerator struct {
	builderID string
	version   string
}

func NewAttestationGenerator(builderID, version string) *AttestationGenerator {
	return &AttestationGenerator{
		builderID: builderID,
		version:   version,
	}
}

func (ag *AttestationGenerator) GenerateAttestation(artifactName, artifactHash string, buildSteps []BuildStep, materials []Material) *SLSAAttestation {
	return &SLSAAttestation{
		PredicateType: "https://slsa.dev/provenance/v0.2",
		Subject: []Subject{
			{
				Name: artifactName,
				Digest: map[string]string{
					"sha256": artifactHash,
				},
			},
		},
		Predicate: Predicate{
			Builder: Builder{
				ID:      ag.builderID,
				Version: ag.version,
			},
			BuildType: "https://shieldx.dev/build/v1",
			Invocation: Invocation{
				ConfigSource: ConfigSource{
					URI: "git+https://github.com/shieldx/build-config",
					Digest: map[string]string{
						"sha256": ag.hashString("build-config"),
					},
				},
			},
			BuildConfig: BuildConfig{
				Steps: buildSteps,
			},
			Materials: materials,
		},
	}
}

func (ag *AttestationGenerator) hashString(s string) string {
	h := sha256.Sum256([]byte(s))
	return string(h[:])
}

type ReproducibleBuild struct {
	SourceHash    string            `json:"source_hash"`
	BuildEnv      map[string]string `json:"build_env"`
	BuildCommands []string          `json:"build_commands"`
	OutputHash    string            `json:"output_hash"`
	Timestamp     time.Time         `json:"timestamp"`
	Attestation   *SLSAAttestation  `json:"attestation"`
}

func (rb *ReproducibleBuild) Verify(expectedHash string) bool {
	return rb.OutputHash == expectedHash
}

func (rb *ReproducibleBuild) ToJSON() ([]byte, error) {
	return json.Marshal(rb)
}