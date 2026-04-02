package judge

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	baseURL    string
	httpClient *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

type ExecuteRequest struct {
	Code         string          `json:"code"`
	FunctionName string          `json:"function_name"`
	Tests        json.RawMessage `json:"tests"`
}

type TestResult struct {
	Name   string  `json:"name"`
	Passed bool    `json:"passed"`
	Time   float64 `json:"time"`
	Error  string  `json:"error,omitempty"`
}

type ExecuteResponse struct {
	Success   bool         `json:"success"`
	Passed    int          `json:"passed"`
	Total     int          `json:"total"`
	TotalTime float64      `json:"total_time"`
	Results   []TestResult `json:"results"`
	Output    string       `json:"output"`
	Error     string       `json:"error,omitempty"`
}

func (c *Client) Execute(req ExecuteRequest) (*ExecuteResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal execute request: %w", err)
	}

	resp, err := c.httpClient.Post(c.baseURL+"/execute", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to call judge service: %w", err)
	}
	defer resp.Body.Close()

	var result ExecuteResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode judge response: %w", err)
	}

	return &result, nil
}
