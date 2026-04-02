package service

import (
	"fmt"

	"gorm.io/gorm"

	"github.com/happytorch/backend/internal/model"
)

type ProblemService struct {
	db *gorm.DB
}

func NewProblemService(db *gorm.DB) *ProblemService {
	return &ProblemService{db: db}
}

type ProblemFilter struct {
	Category   string
	Difficulty string
	Search     string
}

func (s *ProblemService) List(filter ProblemFilter) ([]model.Problem, error) {
	query := s.db.Where("status = ?", "approved").Order("sort_order ASC, created_at ASC")

	if filter.Category != "" {
		query = query.Where("category = ?", filter.Category)
	}
	if filter.Difficulty != "" {
		query = query.Where("difficulty = ?", filter.Difficulty)
	}
	if filter.Search != "" {
		like := "%" + filter.Search + "%"
		query = query.Where("title ILIKE ? OR description ILIKE ?", like, like)
	}

	var problems []model.Problem
	if err := query.Find(&problems).Error; err != nil {
		return nil, fmt.Errorf("failed to list problems: %w", err)
	}
	return problems, nil
}

func (s *ProblemService) GetBySlug(slug string) (*model.Problem, error) {
	var problem model.Problem
	if err := s.db.Where("slug = ? AND status = ?", slug, "approved").First(&problem).Error; err != nil {
		return nil, fmt.Errorf("failed to find problem: %w", err)
	}
	return &problem, nil
}
