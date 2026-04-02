package model

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

type Problem struct {
	ID               uuid.UUID       `gorm:"type:uuid;primaryKey;default:gen_random_uuid()" json:"id"`
	Slug             string          `gorm:"type:varchar(64);uniqueIndex;not null" json:"slug"`
	Title            string          `gorm:"not null" json:"title"`
	Difficulty       string          `gorm:"type:varchar(16);not null" json:"difficulty"`
	Category         string          `gorm:"type:varchar(64);not null" json:"category"`
	FunctionName     string          `gorm:"type:varchar(64)" json:"function_name"`
	Description      string          `gorm:"type:text" json:"description"`
	Signature        string          `json:"signature"`
	Example          string          `gorm:"type:text" json:"example"`
	Hint             string          `gorm:"type:text" json:"hint"`
	TemplateCode     string          `gorm:"type:text" json:"template_code"`
	TestsJSON        json.RawMessage `gorm:"type:jsonb" json:"tests_json"`
	SolutionCode     string          `gorm:"type:text" json:"solution_code,omitempty"`
	SolutionMarkdown string          `gorm:"type:text" json:"solution_markdown,omitempty"`
	Status           string          `gorm:"type:varchar(16);default:'approved'" json:"status"`
	AuthorID         *uuid.UUID      `gorm:"type:uuid" json:"author_id"`
	Author           *User           `gorm:"foreignKey:AuthorID" json:"author,omitempty"`
	ReviewerID       *uuid.UUID      `gorm:"type:uuid" json:"reviewer_id"`
	Reviewer         *User           `gorm:"foreignKey:ReviewerID" json:"reviewer,omitempty"`
	ReviewNote       string          `gorm:"type:text" json:"review_note,omitempty"`
	AIScore          int             `gorm:"default:0" json:"ai_score"`
	SortOrder        int             `gorm:"default:0" json:"sort_order"`
	CreatedAt        time.Time       `json:"created_at"`
	UpdatedAt        time.Time       `json:"updated_at"`
}
