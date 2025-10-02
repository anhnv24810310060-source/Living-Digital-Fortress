package secrets

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type VaultClient struct {
	addr   string
	token  string
	client *http.Client
}

func NewVault() *VaultClient {
	return &VaultClient{
		addr:   strings.TrimRight(os.Getenv("VAULT_ADDR"), "/"),
		token:  os.Getenv("VAULT_TOKEN"),
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

// Get reads KV v2: secret/data/<path>
func (v *VaultClient) Get(ctx context.Context, path string) (map[string]any, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/v1/%s", v.addr, path), nil)
	req.Header.Set("X-Vault-Token", v.token)
	res, err := v.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, fmt.Errorf("vault %s: %s", path, res.Status)
	}
	var body struct {
		Data struct {
			Data map[string]any `json:"data"`
		} `json:"data"`
	}
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		return nil, err
	}
	return body.Data.Data, nil
}
