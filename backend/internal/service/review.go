package service

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/happytorch/backend/internal/judge"
	"github.com/happytorch/backend/internal/llm"
)

// ReviewResult contains the AI review outcome
type ReviewResult struct {
	Passed      bool     `json:"passed"`
	Score       int      `json:"score"`       // 0-100
	Issues      []string `json:"issues"`      // Problems found
	Warnings    []string `json:"warnings"`    // Non-blocking suggestions
	LLMFeedback string   `json:"llm_feedback,omitempty"` // LLM review feedback
	LLMProvider string   `json:"llm_provider,omitempty"` // Which LLM was used
}

// AIReviewService handles automated problem review
type AIReviewService struct {
	judgeClient *judge.Client
	llmProvider llm.Provider
}

func NewAIReviewService(judgeClient *judge.Client) *AIReviewService {
	return &AIReviewService{
		judgeClient: judgeClient,
		llmProvider: &llm.NoopProvider{},
	}
}

// SetLLMProvider configures the LLM provider for enhanced review
func (s *AIReviewService) SetLLMProvider(provider llm.Provider) {
	s.llmProvider = provider
}

// ReviewProblem performs automated review of a submitted problem
func (s *AIReviewService) ReviewProblem(input SubmitProblemInput) (*ReviewResult, error) {
	result := &ReviewResult{
		Passed:   true,
		Score:    100,
		Issues:   []string{},
		Warnings: []string{},
	}

	// 1. Format validation
	s.checkFormat(input, result)

	// 2. Security check
	s.checkSecurity(input, result)

	// 3. Quality check
	s.checkQuality(input, result)

	// 4. Test execution validation
	if err := s.checkTestExecution(input, result); err != nil {
		result.Issues = append(result.Issues, fmt.Sprintf("Test execution error: %v", err))
		result.Passed = false
		result.Score -= 30
	}

	// 5. LLM review (if configured)
	if s.llmProvider.Name() != "none" {
		s.checkWithLLM(input, result)
	}

	// Final score adjustment
	if result.Score < 0 {
		result.Score = 0
	}
	if len(result.Issues) > 0 {
		result.Passed = false
	}

	return result, nil
}

// checkWithLLM uses LLM to evaluate problem quality
func (s *AIReviewService) checkWithLLM(input SubmitProblemInput, result *ReviewResult) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.ReviewRequest{
		Title:        input.Title,
		Description:  input.Description,
		FunctionName: input.FunctionName,
		TemplateCode: input.TemplateCode,
		SolutionCode: input.SolutionCode,
		TestCount:    len(input.Tests),
		Difficulty:   input.Difficulty,
		Category:     input.Category,
	}

	resp, err := s.llmProvider.Review(ctx, req)
	if err != nil {
		result.Warnings = append(result.Warnings, fmt.Sprintf("LLM review unavailable: %v", err))
		return
	}

	result.LLMProvider = s.llmProvider.Name()
	result.LLMFeedback = resp.Feedback

	// Blend LLM score with rule-based score (50/50)
	result.Score = (result.Score + resp.Score) / 2

	if len(resp.Suggestions) > 0 {
		result.Warnings = append(result.Warnings, resp.Suggestions...)
	}

	if !resp.Passed && result.Score < 60 {
		result.Issues = append(result.Issues, "LLM review: quality below threshold")
	}
}

// checkFormat validates problem structure
func (s *AIReviewService) checkFormat(input SubmitProblemInput, result *ReviewResult) {
	// Title length
	if len(input.Title) < 5 {
		result.Issues = append(result.Issues, "Title is too short (minimum 5 characters)")
		result.Score -= 10
	}
	if len(input.Title) > 100 {
		result.Issues = append(result.Issues, "Title is too long (maximum 100 characters)")
		result.Score -= 5
	}

	// Description length
	if len(input.Description) < 50 {
		result.Issues = append(result.Issues, "Description is too short (minimum 50 characters)")
		result.Score -= 15
	}

	// Function name format (snake_case)
	fnPattern := regexp.MustCompile(`^[a-z][a-z0-9_]*$`)
	if !fnPattern.MatchString(input.FunctionName) {
		result.Issues = append(result.Issues, "Function name must be snake_case (e.g., my_function)")
		result.Score -= 10
	}

	// Template code contains function definition
	if !strings.Contains(input.TemplateCode, "def "+input.FunctionName) {
		result.Issues = append(result.Issues, fmt.Sprintf("Template code must define function '%s'", input.FunctionName))
		result.Score -= 20
	}

	// At least 2 test cases recommended
	if len(input.Tests) < 2 {
		result.Warnings = append(result.Warnings, "Consider adding more test cases (recommended: 3+)")
		result.Score -= 5
	}

	// Solution provided
	if input.SolutionCode == "" {
		result.Warnings = append(result.Warnings, "No solution code provided")
		result.Score -= 5
	}
}

// checkSecurity detects potentially malicious code
func (s *AIReviewService) checkSecurity(input SubmitProblemInput, result *ReviewResult) {
	// Dangerous patterns to check in all code fields
	dangerousPatterns := []struct {
		pattern string
		reason  string
	}{
		{`os\.system`, "os.system is not allowed"},
		{`subprocess`, "subprocess module is not allowed"},
		{`eval\s*\(`, "eval() is not allowed"},
		{`exec\s*\(`, "exec() is not allowed"},
		{`__import__\s*\(`, "__import__() is not allowed"},
		{`open\s*\(`, "open() file operations are not allowed"},
		{`requests\.`, "HTTP requests are not allowed"},
		{`urllib`, "URL operations are not allowed"},
		{`socket`, "Socket operations are not allowed"},
		{`pickle`, "pickle module is not allowed"},
		{`globals\s*\(\s*\)`, "globals() is not allowed"},
		{`locals\s*\(\s*\)`, "locals() is not allowed"},
		{`compile\s*\(`, "compile() is not allowed"},
		{`__builtins__`, "Accessing __builtins__ is not allowed"},
	}

	codeToCheck := []string{
		input.TemplateCode,
		input.SolutionCode,
	}
	for _, t := range input.Tests {
		codeToCheck = append(codeToCheck, t.Code)
	}

	allCode := strings.Join(codeToCheck, "\n")

	for _, dp := range dangerousPatterns {
		re := regexp.MustCompile(dp.pattern)
		if re.MatchString(allCode) {
			result.Issues = append(result.Issues, fmt.Sprintf("Security: %s", dp.reason))
			result.Score -= 20
		}
	}
}

// checkQuality evaluates problem quality
func (s *AIReviewService) checkQuality(input SubmitProblemInput, result *ReviewResult) {
	// Check if description contains examples
	if !strings.Contains(strings.ToLower(input.Description), "example") &&
		!strings.Contains(strings.ToLower(input.Description), "signature") {
		result.Warnings = append(result.Warnings, "Description should include examples or function signature")
	}

	// Check if torch is mentioned (it's a PyTorch practice platform)
	allContent := input.Description + input.TemplateCode + input.SolutionCode
	if !strings.Contains(strings.ToLower(allContent), "torch") {
		result.Warnings = append(result.Warnings, "Problem should be related to PyTorch (no torch reference found)")
	}

	// Check test variety
	if len(input.Tests) >= 2 {
		firstTest := input.Tests[0].Code
		allSimilar := true
		for i := 1; i < len(input.Tests); i++ {
			if len(input.Tests[i].Code) != len(firstTest) ||
				strings.Count(input.Tests[i].Code, "torch.") != strings.Count(firstTest, "torch.") {
				allSimilar = false
				break
			}
		}
		if allSimilar {
			result.Warnings = append(result.Warnings, "Test cases appear very similar; consider adding more diverse tests")
		}
	}
}

// checkTestExecution verifies the problem works correctly
func (s *AIReviewService) checkTestExecution(input SubmitProblemInput, result *ReviewResult) error {
	testsJSON, err := json.Marshal(input.Tests)
	if err != nil {
		return fmt.Errorf("failed to marshal tests: %w", err)
	}

	// 1. Template should NOT pass all tests
	templateResult, err := s.judgeClient.Execute(judge.ExecuteRequest{
		Code:         input.TemplateCode,
		FunctionName: input.FunctionName,
		Tests:        testsJSON,
	})
	if err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}
	if templateResult.Success {
		result.Issues = append(result.Issues, "Template code passes all tests (it should not)")
		result.Score -= 25
	}

	// 2. Solution should pass all tests
	if input.SolutionCode != "" {
		solutionResult, err := s.judgeClient.Execute(judge.ExecuteRequest{
			Code:         input.SolutionCode,
			FunctionName: input.FunctionName,
			Tests:        testsJSON,
		})
		if err != nil {
			return fmt.Errorf("failed to execute solution: %w", err)
		}
		if !solutionResult.Success {
			result.Issues = append(result.Issues, fmt.Sprintf("Solution does not pass all tests (%d/%d)", solutionResult.Passed, solutionResult.Total))
			result.Score -= 30
		}
	}

	return nil
}
