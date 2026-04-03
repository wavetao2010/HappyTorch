package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/happytorch/backend/internal/middleware"
	"github.com/happytorch/backend/internal/service"
)

type SubmissionHandler struct {
	submissionService *service.SubmissionService
	problemService    *service.ProblemService
}

func NewSubmissionHandler(submissionService *service.SubmissionService, problemService *service.ProblemService) *SubmissionHandler {
	return &SubmissionHandler{
		submissionService: submissionService,
		problemService:    problemService,
	}
}

type submitRequest struct {
	Code string `json:"code" binding:"required"`
}

func (h *SubmissionHandler) Submit(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)
	slug := c.Param("slug")

	var req submitRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "code is required"})
		return
	}

	submission, result, err := h.submissionService.SubmitCode(userID, slug, req.Code)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to process submission"})
		return
	}

	testResults := make([]gin.H, len(result.Results))
	for i, r := range result.Results {
		testResults[i] = gin.H{
			"name":    r.Name,
			"passed":  r.Passed,
			"message": r.Error,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"result": gin.H{
			"submission_id":  submission.ID,
			"passed":         result.Success,
			"output":         result.Output,
			"error":          result.Error,
			"test_results":   testResults,
			"execution_time": result.TotalTime,
		},
	})
}

func (h *SubmissionHandler) GetSolution(c *gin.Context) {
	slug := c.Param("slug")

	problem, err := h.problemService.GetBySlug(slug)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "problem not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"solution":      problem.SolutionMarkdown,
		"solution_code": problem.SolutionCode,
	})
}
