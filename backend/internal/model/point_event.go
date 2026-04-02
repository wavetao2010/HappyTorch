package model

import (
	"time"

	"github.com/google/uuid"
)

type PointEvent struct {
	ID          uuid.UUID `gorm:"type:uuid;primaryKey;default:gen_random_uuid()" json:"id"`
	UserID      uuid.UUID `gorm:"type:uuid;not null" json:"user_id"`
	User        *User     `gorm:"foreignKey:UserID" json:"user,omitempty"`
	EventType   string    `gorm:"type:varchar(32);not null" json:"event_type"`
	Points      int       `gorm:"not null" json:"points"`
	ReferenceID uuid.UUID `gorm:"type:uuid" json:"reference_id"`
	CreatedAt   time.Time `json:"created_at"`
}
