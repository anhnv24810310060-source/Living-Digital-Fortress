package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type CreditsClient struct {
	baseURL    string
	httpClient *http.Client
}

type ConsumeRequest struct {
	TenantID       string `json:"tenant_id"`
	Amount         int64  `json:"amount"`
	Description    string `json:"description"`
	Reference      string `json:"reference"`
	IdempotencyKey string `json:"idempotency_key"`
}

type CreditResponse struct {
	Success       bool   `json:"success"`
	TransactionID string `json:"transaction_id,omitempty"`
	Balance       int64  `json:"balance,omitempty"`
	Message       string `json:"message,omitempty"`
	Error         string `json:"error,omitempty"`
}

type BalanceResponse struct {
	TenantID string `json:"tenant_id"`
	Balance  int64  `json:"balance"`
}

func NewCreditsClient(baseURL string) *CreditsClient {
	return &CreditsClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func (c *CreditsClient) ConsumeCredits(req ConsumeRequest) error {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/credits/consume", c.baseURL)
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to consume credits: %w", err)
	}
	defer resp.Body.Close()

	var response CreditResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	if !response.Success {
		return fmt.Errorf("credit consumption failed: %s", response.Error)
	}

	return nil
}

func (c *CreditsClient) HasSufficientCredits(tenantID string, amount int64) (bool, error) {
	balance, err := c.GetBalance(tenantID)
	if err != nil {
		return false, err
	}
	return balance >= amount, nil
}

func (c *CreditsClient) GetBalance(tenantID string) (int64, error) {
	url := fmt.Sprintf("%s/credits/balance/%s", c.baseURL, tenantID)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return 0, fmt.Errorf("failed to get balance: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("credits service returned status %d", resp.StatusCode)
	}

	var response BalanceResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return 0, fmt.Errorf("failed to decode balance response: %w", err)
	}

	return response.Balance, nil
}

func (c *CreditsClient) CheckHealth() error {
	url := fmt.Sprintf("%s/health", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return fmt.Errorf("credits health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("credits service unhealthy: status %d", resp.StatusCode)
	}

	return nil
}
