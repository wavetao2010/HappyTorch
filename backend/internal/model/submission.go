package model

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

type Submission struct {
	ID          uuid.UUID       `gorm:"type:uuid;primaryKey;default:gen_random_uuid()" json:"id"`
	UserID      uuid.UUID       `gorm:"type:uuid;not null" json:"user_id"`
	User        *User           `gorm:"foreignKey:UserID" json:"user,omitempty"`
	ProblemID   uuid.UUID       `gorm:"type:uuid;not null" json:"problem_id"`
	Problem     *Problem        `gorm:"foreignKey:ProblemID" json:"problem,omitempty"`
	Code        string          `gorm:"type:text;not null" json:"code"`
	Passed      int             `json:"passed"`
	Total       int             `json:"total"`
	TotalTime   float64         `json:"total_time"`
	Success     bool            `json:"success"`
	ResultsJSON json.RawMessage `gorm:"type:jsonb" json:"results_json"`
	Output      string          `gorm:"type:text" json:"output"`
	CreatedAt   time.Time       `json:"created_at"`
}
