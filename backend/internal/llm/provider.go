package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ReviewRequest is the input for LLM review
type ReviewRequest struct {
	Title        string `json:"title"`
	Description  string `json:"description"`
	FunctionName string `json:"function_name"`
	TemplateCode string `json:"template_code"`
	SolutionCode string `json:"solution_code"`
	TestCount    int    `json:"test_count"`
	Difficulty   string `json:"difficulty"`
	Category     string `json:"category"`
}

// ReviewResponse is the LLM review output
type ReviewResponse struct {
	Score       int      `json:"score"`        // 0-100
	Passed      bool     `json:"passed"`       // score >= threshold
	Feedback    string   `json:"feedback"`     // Summary feedback
	Suggestions []string `json:"suggestions"`  // Improvement suggestions
}

// Provider interface for LLM providers
type Provider interface {
	Name() string
	Review(ctx context.Context, req ReviewRequest) (*ReviewResponse, error)
}

// Config holds LLM configuration
type Config struct {
	Provider       string // gemini, qwen, claude, openai, none
	APIKey         string
	Model          string // optional model override
	ScoreThreshold int    // minimum score to pass (default 70)
}

// NewProvider creates a provider based on config
func NewProvider(cfg Config) Provider {
	if cfg.ScoreThreshold == 0 {
		cfg.ScoreThreshold = 70
	}

	switch cfg.Provider {
	case "gemini":
		return &GeminiProvider{apiKey: cfg.APIKey, model: cfg.Model, threshold: cfg.ScoreThreshold}
	case "qwen":
		return &QwenProvider{apiKey: cfg.APIKey, model: cfg.Model, threshold: cfg.ScoreThreshold}
	case "claude":
		return &ClaudeProvider{apiKey: cfg.APIKey, model: cfg.Model, threshold: cfg.ScoreThreshold}
	case "openai":
		return &OpenAIProvider{apiKey: cfg.APIKey, model: cfg.Model, threshold: cfg.ScoreThreshold}
	default:
		return &NoopProvider{}
	}
}

// NoopProvider does nothing (LLM disabled)
type NoopProvider struct{}

func (p *NoopProvider) Name() string { return "none" }
func (p *NoopProvider) Review(ctx context.Context, req ReviewRequest) (*ReviewResponse, error) {
	return &ReviewResponse{Score: 100, Passed: true, Feedback: "LLM review disabled"}, nil
}

// buildPrompt creates the review prompt
func buildPrompt(req ReviewRequest) string {
	return fmt.Sprintf(`You are reviewing a PyTorch coding problem submission for a practice platform.

**Problem Details:**
- Title: %s
- Difficulty: %s
- Category: %s
- Function Name: %s
- Number of Test Cases: %d

**Description:**
%s

**Template Code:**
%s

**Solution Code:**
%s

**Review Criteria:**
1. Is the problem description clear and well-written?
2. Is this a valid PyTorch exercise (uses torch operations)?
3. Is the difficulty rating appropriate?
4. Are there enough test cases?
5. Is the solution code correct and idiomatic?

**Respond in JSON format only:**
{
  "score": <0-100>,
  "feedback": "<one paragraph summary>",
  "suggestions": ["<suggestion 1>", "<suggestion 2>"]
}`,
		req.Title, req.Difficulty, req.Category, req.FunctionName, req.TestCount,
		req.Description, req.TemplateCode, req.SolutionCode)
}

// parseJSONResponse extracts JSON from LLM response
func parseJSONResponse(body string, threshold int) (*ReviewResponse, error) {
	// Try to find JSON in response
	start := -1
	end := -1
	braceCount := 0

	for i, c := range body {
		if c == '{' {
			if start == -1 {
				start = i
			}
			braceCount++
		} else if c == '}' {
			braceCount--
			if braceCount == 0 && start != -1 {
				end = i + 1
				break
			}
		}
	}

	if start == -1 || end == -1 {
		return nil, fmt.Errorf("no JSON found in response")
	}

	var resp ReviewResponse
	if err := json.Unmarshal([]byte(body[start:end]), &resp); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	resp.Passed = resp.Score >= threshold
	return &resp, nil
}

// --- Gemini Provider ---

type GeminiProvider struct {
	apiKey    string
	model     string
	threshold int
}

func (p *GeminiProvider) Name() string { return "gemini" }

func (p *GeminiProvider) Review(ctx context.Context, req ReviewRequest) (*ReviewResponse, error) {
	model := p.model
	if model == "" {
		model = "gemini-1.5-flash"
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", model, p.apiKey)

	payload := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]string{
					{"text": buildPrompt(req)},
				},
			},
		},
		"generationConfig": map[string]interface{}{
			"temperature": 0.3,
			"maxOutputTokens": 1024,
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	var result struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini response: %w", err)
	}

	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("empty Gemini response")
	}

	return parseJSONResponse(result.Candidates[0].Content.Parts[0].Text, p.threshold)
}

// --- Qwen Provider ---

type QwenProvider struct {
	apiKey    string
	model     string
	threshold int
}

func (p *QwenProvider) Name() string { return "qwen" }

func (p *QwenProvider) Review(ctx context.Context, req ReviewRequest) (*ReviewResponse, error) {
	model := p.model
	if model == "" {
		model = "qwen-turbo"
	}

	url := "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

	payload := map[string]interface{}{
		"model": model,
		"input": map[string]interface{}{
			"messages": []map[string]string{
				{"role": "user", "content": buildPrompt(req)},
			},
		},
		"parameters": map[string]interface{}{
			"temperature": 0.3,
			"max_tokens":  1024,
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	var result struct {
		Output struct {
			Text string `json:"text"`
		} `json:"output"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse Qwen response: %w", err)
	}

	return parseJSONResponse(result.Output.Text, p.threshold)
}

// --- Claude Provider ---

type ClaudeProvider struct {
	apiKey    string
	model     string
	threshold int
}

func (p *ClaudeProvider) Name() string { return "claude" }

func (p *ClaudeProvider) Review(ctx context.Context, req ReviewRequest) (*ReviewResponse, error) {
	model := p.model
	if model == "" {
		model = "claude-3-haiku-20240307"
	}

	url := "https://api.anthropic.com/v1/messages"

	payload := map[string]interface{}{
		"model":      model,
		"max_tokens": 1024,
		"messages": []map[string]string{
			{"role": "user", "content": buildPrompt(req)},
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	var result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse Claude response: %w", err)
	}

	if len(result.Content) == 0 {
		return nil, fmt.Errorf("empty Claude response")
	}

	return parseJSONResponse(result.Content[0].Text, p.threshold)
}

// --- OpenAI Provider ---

type OpenAIProvider struct {
	apiKey    string
	model     string
	threshold int
}

func (p *OpenAIProvider) Name() string { return "openai" }

func (p *OpenAIProvider) Review(ctx context.Context, req ReviewRequest) (*ReviewResponse, error) {
	model := p.model
	if model == "" {
		model = "gpt-4o-mini"
	}

	url := "https://api.openai.com/v1/chat/completions"

	payload := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": buildPrompt(req)},
		},
		"temperature": 0.3,
		"max_tokens":  1024,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI response: %w", err)
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("empty OpenAI response")
	}

	return parseJSONResponse(result.Choices[0].Message.Content, p.threshold)
}
