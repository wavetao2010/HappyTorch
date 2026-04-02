package handler

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/happytorch/backend/internal/middleware"
	"github.com/happytorch/backend/internal/service"
)

type ProgressHandler struct {
	submissionService *service.SubmissionService
}

func NewProgressHandler(submissionService *service.SubmissionService) *ProgressHandler {
	return &ProgressHandler{submissionService: submissionService}
}

func (h *ProgressHandler) GetProgress(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	entries, err := h.submissionService.GetProgress(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load progress"})
		return
	}

	var solvedSlugs []string
	var attemptedSlugs []string
	for _, e := range entries {
		if e.Solved {
			solvedSlugs = append(solvedSlugs, e.Slug)
		} else {
			attemptedSlugs = append(attemptedSlugs, e.Slug)
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"progress": gin.H{
			"solved_slugs":    solvedSlugs,
			"attempted_slugs": attemptedSlugs,
		},
	})
}

func (h *ProgressHandler) GetSubmissions(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))

	submissions, total, err := h.submissionService.GetSubmissions(userID, page, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load submissions"})
		return
	}

	// Flatten submission data for frontend
	result := make([]gin.H, len(submissions))
	for i, s := range submissions {
		problemSlug := ""
		problemTitle := ""
		if s.Problem != nil {
			problemSlug = s.Problem.Slug
			problemTitle = s.Problem.Title
		}
		result[i] = gin.H{
			"id":             s.ID,
			"problem_slug":   problemSlug,
			"problem_title":  problemTitle,
			"code":           s.Code,
			"passed":         s.Passed,
			"total":          s.Total,
			"success":        s.Success,
			"execution_time": s.TotalTime,
			"created_at":     s.CreatedAt,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"submissions": result,
		"total":       total,
		"page":        page,
		"limit":       limit,
	})
}
