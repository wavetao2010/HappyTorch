package handler

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/happytorch/backend/internal/middleware"
	"github.com/happytorch/backend/internal/service"
)

type CommunityHandler struct {
	communityService *service.CommunityService
}

func NewCommunityHandler(communityService *service.CommunityService) *CommunityHandler {
	return &CommunityHandler{communityService: communityService}
}

// POST /api/problems/submit
func (h *CommunityHandler) SubmitProblem(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	var input service.SubmitProblemInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := h.communityService.SubmitProblem(userID, input)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// If AI review rejected, return review result with 422 status
	if result.Problem == nil {
		c.JSON(http.StatusUnprocessableEntity, gin.H{
			"error":  "AI review failed",
			"review": result.Review,
		})
		return
	}

	c.JSON(http.StatusCreated, result)
}

// GET /api/problems/mine
func (h *CommunityHandler) ListMyProblems(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	problems, err := h.communityService.ListMyProblems(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load problems"})
		return
	}

	c.JSON(http.StatusOK, problems)
}

// PUT /api/problems/:id
func (h *CommunityHandler) UpdateMyProblem(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	problemID, err := uuid.Parse(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid problem ID"})
		return
	}

	var input service.SubmitProblemInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	problem, err := h.communityService.UpdateMyProblem(userID, problemID, input)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, problem)
}

// GET /api/admin/problems/pending
func (h *CommunityHandler) ListPendingProblems(c *gin.Context) {
	problems, err := h.communityService.ListPendingProblems()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load pending problems"})
		return
	}

	c.JSON(http.StatusOK, problems)
}

// GET /api/admin/problems/:id
func (h *CommunityHandler) GetPendingProblem(c *gin.Context) {
	problemID, err := uuid.Parse(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid problem ID"})
		return
	}

	problem, err := h.communityService.GetPendingProblem(problemID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "problem not found"})
		return
	}

	c.JSON(http.StatusOK, problem)
}

// POST /api/admin/problems/:id/review
func (h *CommunityHandler) ReviewProblem(c *gin.Context) {
	reviewerID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	problemID, err := uuid.Parse(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid problem ID"})
		return
	}

	var input service.ReviewInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	problem, err := h.communityService.ReviewProblem(reviewerID, problemID, input)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, problem)
}

// GET /api/admin/users
func (h *CommunityHandler) ListUsers(c *gin.Context) {
	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))

	users, total, err := h.communityService.ListUsers(page, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load users"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"users": users,
		"total": total,
		"page":  page,
		"limit": limit,
	})
}

// PUT /api/admin/users/:id/role
func (h *CommunityHandler) ChangeUserRole(c *gin.Context) {
	userID, err := uuid.Parse(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user ID"})
		return
	}

	var input service.ChangeRoleInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	user, err := h.communityService.ChangeUserRole(userID, input)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, user)
}

// GET /api/leaderboard
func (h *CommunityHandler) GetLeaderboard(c *gin.Context) {
	sortBy := c.DefaultQuery("sort", "points")
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "50"))

	entries, err := h.communityService.GetLeaderboard(sortBy, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load leaderboard"})
		return
	}

	c.JSON(http.StatusOK, entries)
}

// GET /api/users/:username
func (h *CommunityHandler) GetUserProfile(c *gin.Context) {
	username := c.Param("username")

	profile, err := h.communityService.GetUserProfile(username)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "user not found"})
		return
	}

	c.JSON(http.StatusOK, profile)
}

// --- Role Applications ---

// POST /api/applications/moderator
func (h *CommunityHandler) ApplyForModerator(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	var input service.ApplyModeratorInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	app, err := h.communityService.ApplyForModerator(userID, input)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, app)
}

// GET /api/applications/mine
func (h *CommunityHandler) GetMyApplications(c *gin.Context) {
	userID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	apps, err := h.communityService.GetMyApplications(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get applications"})
		return
	}

	c.JSON(http.StatusOK, apps)
}

// GET /api/admin/applications
func (h *CommunityHandler) ListRoleApplications(c *gin.Context) {
	apps, err := h.communityService.ListRoleApplications()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to list applications"})
		return
	}

	c.JSON(http.StatusOK, apps)
}

// POST /api/admin/applications/:id/review
func (h *CommunityHandler) ReviewApplication(c *gin.Context) {
	reviewerID := c.MustGet(middleware.ContextUserID).(uuid.UUID)

	appID, err := uuid.Parse(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid application ID"})
		return
	}

	var input service.ReviewApplicationInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	app, err := h.communityService.ReviewApplication(reviewerID, appID, input)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, app)
}
