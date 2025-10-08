package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/open-policy-agent/opa/rego"
)

var (
	ErrAccessDenied     = errors.New("access denied")
	ErrInvalidPolicy    = errors.New("invalid policy")
	ErrPolicyEvaluation = errors.New("policy evaluation failed")
)

// Permission represents a granular permission
type Permission struct {
	Resource string   `json:"resource"`
	Actions  []string `json:"actions"`
}

// Role represents a role with permissions
type Role struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Permissions []Permission `json:"permissions"`
	Inherits    []string     `json:"inherits,omitempty"`
}

// RBACEngine handles Role-Based Access Control with OPA
type RBACEngine struct {
	policies map[string]*rego.PreparedEvalQuery
	roles    map[string]*Role
}

// RBACConfig configuration for RBAC engine
type RBACConfig struct {
	PolicyPath string
	Roles      []*Role
}

// NewRBACEngine creates a new RBAC engine with OPA policies
func NewRBACEngine(config RBACConfig) (*RBACEngine, error) {
	engine := &RBACEngine{
		policies: make(map[string]*rego.PreparedEvalQuery),
		roles:    make(map[string]*Role),
	}

	// Load roles
	for _, role := range config.Roles {
		engine.roles[role.Name] = role
	}

	// Load default policies if no custom path provided
	if config.PolicyPath == "" {
		if err := engine.loadDefaultPolicies(); err != nil {
			return nil, fmt.Errorf("failed to load default policies: %w", err)
		}
	}

	return engine, nil
}

// CheckPermission checks if user has permission to perform action on resource
func (re *RBACEngine) CheckPermission(ctx context.Context, userRoles []string, resource, action string) (bool, error) {
	// Collect all permissions from user roles
	permissions := re.collectPermissions(userRoles)

	// Check if permission exists
	for _, perm := range permissions {
		if perm.Resource == resource || perm.Resource == "*" {
			for _, a := range perm.Actions {
				if a == action || a == "*" {
					return true, nil
				}
			}
		}
	}

	return false, nil
}

// CheckPermissionWithPolicy checks permission using OPA policy
func (re *RBACEngine) CheckPermissionWithPolicy(ctx context.Context, policyName string, input map[string]interface{}) (bool, error) {
	query, exists := re.policies[policyName]
	if !exists {
		return false, fmt.Errorf("%w: policy %s not found", ErrInvalidPolicy, policyName)
	}

	results, err := query.Eval(ctx, rego.EvalInput(input))
	if err != nil {
		return false, fmt.Errorf("%w: %v", ErrPolicyEvaluation, err)
	}

	if len(results) == 0 {
		return false, nil
	}

	// Check if result is allowed
	if allowed, ok := results[0].Expressions[0].Value.(bool); ok {
		return allowed, nil
	}

	return false, nil
}

// LoadPolicy loads an OPA policy
func (re *RBACEngine) LoadPolicy(ctx context.Context, name, module string) error {
	query, err := rego.New(
		rego.Query("data.shieldx."+name+".allow"),
		rego.Module(name+".rego", module),
	).PrepareForEval(ctx)

	if err != nil {
		return fmt.Errorf("failed to prepare policy: %w", err)
	}

	re.policies[name] = &query
	return nil
}

// AddRole adds or updates a role
func (re *RBACEngine) AddRole(role *Role) {
	re.roles[role.Name] = role
}

// GetRole retrieves a role by name
func (re *RBACEngine) GetRole(name string) (*Role, bool) {
	role, exists := re.roles[name]
	return role, exists
}

// collectPermissions collects all permissions from roles including inherited roles
func (re *RBACEngine) collectPermissions(roleNames []string) []Permission {
	visited := make(map[string]bool)
	var permissions []Permission

	var collect func(roleName string)
	collect = func(roleName string) {
		if visited[roleName] {
			return
		}
		visited[roleName] = true

		role, exists := re.roles[roleName]
		if !exists {
			return
		}

		permissions = append(permissions, role.Permissions...)

		// Process inherited roles
		for _, inherited := range role.Inherits {
			collect(inherited)
		}
	}

	for _, roleName := range roleNames {
		collect(roleName)
	}

	return permissions
}

// loadDefaultPolicies loads default OPA policies
func (re *RBACEngine) loadDefaultPolicies() error {
	ctx := context.Background()

	// API Access Policy
	apiPolicy := `
package shieldx.api_access

default allow = false

# Admin has full access
allow {
	input.roles[_] == "admin"
}

# Service accounts can access their own resources
allow {
	input.roles[_] == "service"
	input.resource_tenant_id == input.user_tenant_id
}

# Users can access their own resources
allow {
	input.roles[_] == "user"
	input.action == "read"
	input.resource_owner_id == input.user_id
}

# Users can update their own resources
allow {
	input.roles[_] == "user"
	input.action == "update"
	input.resource_owner_id == input.user_id
}
`
	if err := re.LoadPolicy(ctx, "api_access", apiPolicy); err != nil {
		return err
	}

	// Data Access Policy
	dataPolicy := `
package shieldx.data_access

default allow = false

# Admin can access all data
allow {
	input.roles[_] == "admin"
}

# Users can access data in their tenant
allow {
	input.roles[_] == "user"
	input.resource_tenant_id == input.user_tenant_id
	data_sensitivity_allowed
}

# Check data sensitivity level
data_sensitivity_allowed {
	input.data_sensitivity == "public"
}

data_sensitivity_allowed {
	input.data_sensitivity == "internal"
	input.roles[_] == "user"
}

data_sensitivity_allowed {
	input.data_sensitivity == "confidential"
	input.roles[_] == "manager"
}
`
	if err := re.LoadPolicy(ctx, "data_access", dataPolicy); err != nil {
		return err
	}

	return nil
}

// Default roles
func GetDefaultRoles() []*Role {
	return []*Role{
		{
			Name:        "admin",
			Description: "Full system access",
			Permissions: []Permission{
				{Resource: "*", Actions: []string{"*"}},
			},
		},
		{
			Name:        "user",
			Description: "Standard user access",
			Permissions: []Permission{
				{Resource: "profile", Actions: []string{"read", "update"}},
				{Resource: "dashboard", Actions: []string{"read"}},
				{Resource: "api", Actions: []string{"read"}},
			},
		},
		{
			Name:        "service",
			Description: "Service account access",
			Permissions: []Permission{
				{Resource: "api", Actions: []string{"read", "write"}},
				{Resource: "metrics", Actions: []string{"write"}},
			},
		},
		{
			Name:        "auditor",
			Description: "Read-only audit access",
			Permissions: []Permission{
				{Resource: "logs", Actions: []string{"read"}},
				{Resource: "audit", Actions: []string{"read"}},
				{Resource: "metrics", Actions: []string{"read"}},
			},
		},
		{
			Name:        "operator",
			Description: "Operational access",
			Permissions: []Permission{
				{Resource: "services", Actions: []string{"read", "restart"}},
				{Resource: "config", Actions: []string{"read"}},
				{Resource: "metrics", Actions: []string{"read"}},
			},
			Inherits: []string{"user"},
		},
	}
}

// MarshalRoles converts roles to JSON
func MarshalRoles(roles []*Role) (string, error) {
	data, err := json.MarshalIndent(roles, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// UnmarshalRoles parses roles from JSON
func UnmarshalRoles(data string) ([]*Role, error) {
	var roles []*Role
	err := json.Unmarshal([]byte(data), &roles)
	if err != nil {
		return nil, err
	}
	return roles, nil
}
