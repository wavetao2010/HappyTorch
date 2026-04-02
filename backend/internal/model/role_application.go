package model

import (
	"time"

	"github.com/google/uuid"
)

// RoleApplication tracks moderator role applications
type RoleApplication struct {
	ID          uuid.UUID  `gorm:"type:uuid;primaryKey;default:gen_random_uuid()" json:"id"`
	UserID      uuid.UUID  `gorm:"type:uuid;not null" json:"user_id"`
	User        *User      `gorm:"foreignKey:UserID" json:"user,omitempty"`
	Role        string     `gorm:"type:varchar(16);not null" json:"role"` // moderator
	Status      string     `gorm:"type:varchar(16);default:'pending'" json:"status"` // pending, approved, rejected
	Reason      string     `gorm:"type:text" json:"reason"` // Why user wants the role
	ReviewerID  *uuid.UUID `gorm:"type:uuid" json:"reviewer_id"`
	Reviewer    *User      `gorm:"foreignKey:ReviewerID" json:"reviewer,omitempty"`
	ReviewNote  string     `gorm:"type:text" json:"review_note,omitempty"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}
