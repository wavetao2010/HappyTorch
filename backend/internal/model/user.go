package model

import (
	"time"

	"github.com/google/uuid"
)

type User struct {
	ID             uuid.UUID `gorm:"type:uuid;primaryKey;default:gen_random_uuid()" json:"id"`
	Username       string    `gorm:"type:varchar(32);uniqueIndex;not null" json:"username"`
	Email          string    `gorm:"type:varchar(255);uniqueIndex;not null" json:"email"`
	PasswordHash   string    `json:"-"`
	GitHubID       int64     `gorm:"uniqueIndex" json:"-"`
	DisplayName    string    `gorm:"type:varchar(64)" json:"display_name"`
	AvatarURL      string    `json:"avatar_url"`
	Role           string    `gorm:"type:varchar(16);default:'user'" json:"role"`
	Points         int       `gorm:"default:0" json:"points"`
	ProblemsSolved int       `gorm:"default:0" json:"problems_solved"`
	Bio            string    `json:"bio"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}
