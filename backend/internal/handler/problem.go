package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/happytorch/backend/internal/service"
)

type ProblemHandler struct {
	problemService *service.ProblemService
}

func NewProblemHandler(problemService *service.ProblemService) *ProblemHandler {
	return &ProblemHandler{problemService: problemService}
}

func (h *ProblemHandler) List(c *gin.Context) {
	filter := service.ProblemFilter{
		Category:   c.Query("category"),
		Difficulty: c.Query("difficulty"),
		Search:     c.Query("search"),
	}

	problems, err := h.problemService.List(filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load problems"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"problems": problems})
}

func (h *ProblemHandler) GetBySlug(c *gin.Context) {
	slug := c.Param("slug")

	problem, err := h.problemService.GetBySlug(slug)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "problem not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"problem": problem})
}
