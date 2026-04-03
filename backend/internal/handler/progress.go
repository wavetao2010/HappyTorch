package handler

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gorm.io/gorm"

	"github.com/happytorch/backend/internal/middleware"
	"github.com/happytorch/backend/internal/model"
	"github.com/happytorch/backend/internal/service"
)

type ProgressHandler struct {
	db                *gorm.DB
	submissionService *service.SubmissionService
}

func NewProgressHandler(db *gorm.DB, submissionService *service.SubmissionService) *ProgressHandler {
	return &ProgressHandler{db: db, submissionService: submissionService}
}

func (h *ProgressHandler) GetProgress(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	// Get user info (points, problems_solved)
	var user model.User
	if err := h.db.First(&user, "id = ?", userID).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load user"})
		return
	}

	// Calculate rank
	var rank int64
	h.db.Model(&model.User{}).Where("points > ?", user.Points).Count(&rank)
	rank++ // 1-indexed

	// Get problem counts by difficulty
	type diffCount struct {
		Difficulty string
		Total      int
	}
	var diffCounts []diffCount
	h.db.Model(&model.Problem{}).
		Select("difficulty, COUNT(*) as total").
		Where("status = ?", "approved").
		Group("difficulty").
		Scan(&diffCounts)

	// Get solved counts by difficulty for this user
	type solvedCount struct {
		Difficulty string
		Solved     int
	}
	var solvedCounts []solvedCount
	h.db.Raw(`
		SELECT p.difficulty, COUNT(DISTINCT p.id) as solved
		FROM submissions s
		JOIN problems p ON p.id = s.problem_id
		WHERE s.user_id = ? AND s.success = true AND p.status = 'approved'
		GROUP BY p.difficulty
	`, userID).Scan(&solvedCounts)

	// Build by_difficulty map
	byDifficulty := map[string]gin.H{
		"easy":   {"solved": 0, "total": 0},
		"medium": {"solved": 0, "total": 0},
		"hard":   {"solved": 0, "total": 0},
	}
	for _, dc := range diffCounts {
		key := strings.ToLower(dc.Difficulty)
		if _, ok := byDifficulty[key]; ok {
			byDifficulty[key]["total"] = dc.Total
		}
	}
	for _, sc := range solvedCounts {
		key := strings.ToLower(sc.Difficulty)
		if _, ok := byDifficulty[key]; ok {
			byDifficulty[key]["solved"] = sc.Solved
		}
	}

	// Also include solved/attempted slugs for problem list highlighting
	entries, _ := h.submissionService.GetProgress(userID)
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
			"points":          user.Points,
			"solved":          user.ProblemsSolved,
			"rank":            rank,
			"by_difficulty":   byDifficulty,
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
