package service

import (
	"fmt"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"github.com/happytorch/backend/internal/model"
)

type PointService struct {
	db *gorm.DB
}

func NewPointService(db *gorm.DB) *PointService {
	return &PointService{db: db}
}

func (s *PointService) AwardPoints(userID uuid.UUID, eventType string, points int, referenceID uuid.UUID) error {
	event := model.PointEvent{
		UserID:      userID,
		EventType:   eventType,
		Points:      points,
		ReferenceID: referenceID,
	}

	return s.db.Transaction(func(tx *gorm.DB) error {
		if err := tx.Create(&event).Error; err != nil {
			return fmt.Errorf("failed to create point event: %w", err)
		}

		if err := tx.Model(&model.User{}).Where("id = ?", userID).
			Update("points", gorm.Expr("points + ?", points)).Error; err != nil {
			return fmt.Errorf("failed to update user points: %w", err)
		}

		return nil
	})
}

// CheckAndPromoteRole promotes user to contributor if they have approved problems
// Note: moderator requires application, admin is manual-only
func (s *PointService) CheckAndPromoteRole(userID uuid.UUID) error {
	var user model.User
	if err := s.db.First(&user, "id = ?", userID).Error; err != nil {
		return fmt.Errorf("failed to find user for promotion check: %w", err)
	}

	// Don't downgrade higher roles
	if user.Role == "moderator" || user.Role == "admin" {
		return nil
	}

	// Check if user has any approved problems -> contributor
	var approvedCount int64
	s.db.Model(&model.Problem{}).Where("author_id = ? AND status = ?", userID, "approved").Count(&approvedCount)

	if approvedCount > 0 && user.Role == "user" {
		if err := s.db.Model(&user).Update("role", "contributor").Error; err != nil {
			return fmt.Errorf("failed to promote user to contributor: %w", err)
		}
	}

	return nil
}

// CanApplyForModerator checks if user is eligible to apply for moderator role
func (s *PointService) CanApplyForModerator(userID uuid.UUID) (bool, string) {
	var user model.User
	if err := s.db.First(&user, "id = ?", userID).Error; err != nil {
		return false, "User not found"
	}

	if user.Role == "moderator" || user.Role == "admin" {
		return false, "Already a moderator or admin"
	}

	if user.Points < 500 {
		return false, fmt.Sprintf("Need at least 500 points (current: %d)", user.Points)
	}

	// Check if already has pending application
	var pending model.RoleApplication
	err := s.db.Where("user_id = ? AND status = ?", userID, "pending").First(&pending).Error
	if err == nil {
		return false, "Already have a pending application"
	}

	return true, ""
}
