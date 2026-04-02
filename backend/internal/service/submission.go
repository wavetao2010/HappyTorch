package service

import (
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"github.com/happytorch/backend/internal/judge"
	"github.com/happytorch/backend/internal/model"
)

type SubmissionService struct {
	db           *gorm.DB
	judgeClient  *judge.Client
	pointService *PointService
	cacheService *CacheService
}

func NewSubmissionService(db *gorm.DB, judgeClient *judge.Client, pointService *PointService, cacheService *CacheService) *SubmissionService {
	return &SubmissionService{
		db:           db,
		judgeClient:  judgeClient,
		pointService: pointService,
		cacheService: cacheService,
	}
}

func (s *SubmissionService) SubmitCode(userID uuid.UUID, slug, code string) (*model.Submission, *judge.ExecuteResponse, error) {
	var problem model.Problem
	if err := s.db.Where("slug = ? AND status = ?", slug, "approved").First(&problem).Error; err != nil {
		return nil, nil, fmt.Errorf("failed to find problem: %w", err)
	}

	result, err := s.judgeClient.Execute(judge.ExecuteRequest{
		Code:         code,
		FunctionName: problem.FunctionName,
		Tests:        problem.TestsJSON,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute code: %w", err)
	}

	resultsJSON, err := json.Marshal(result.Results)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal results: %w", err)
	}

	// Check if already solved before saving the new submission
	var alreadySolved bool
	if result.Success {
		alreadySolved, _ = s.HasSolved(userID, problem.ID)
	}

	submission := model.Submission{
		UserID:      userID,
		ProblemID:   problem.ID,
		Code:        code,
		Passed:      result.Passed,
		Total:       result.Total,
		TotalTime:   result.TotalTime,
		Success:     result.Success,
		ResultsJSON: resultsJSON,
		Output:      result.Output,
	}

	if err := s.db.Create(&submission).Error; err != nil {
		return nil, nil, fmt.Errorf("failed to save submission: %w", err)
	}

	if result.Success && !alreadySolved {
		points := 10 + difficultyBonus(problem.Difficulty)
		if err := s.pointService.AwardPoints(userID, "problem_solved", points, submission.ID); err != nil {
			return &submission, result, nil
		}
		if err := s.db.Model(&model.User{}).Where("id = ?", userID).
			Update("problems_solved", gorm.Expr("problems_solved + 1")).Error; err != nil {
			return &submission, result, nil
		}
		_ = s.pointService.CheckAndPromoteRole(userID)
		if s.cacheService != nil {
			s.cacheService.InvalidateLeaderboard()
		}
	}

	return &submission, result, nil
}

func (s *SubmissionService) HasSolved(userID, problemID uuid.UUID) (bool, error) {
	var count int64
	err := s.db.Model(&model.Submission{}).
		Where("user_id = ? AND problem_id = ? AND success = ?", userID, problemID, true).
		Count(&count).Error
	if err != nil {
		return false, fmt.Errorf("failed to check solved status: %w", err)
	}
	return count > 0, nil
}

type ProgressEntry struct {
	ProblemID uuid.UUID `json:"problem_id"`
	Slug      string    `json:"slug"`
	Solved    bool      `json:"solved"`
	Attempts  int       `json:"attempts"`
	BestTime  float64   `json:"best_time"`
}

func (s *SubmissionService) GetProgress(userID uuid.UUID) ([]ProgressEntry, error) {
	type row struct {
		ProblemID  uuid.UUID
		Slug       string
		Attempts   int
		HasSuccess bool
		BestTime   float64
	}

	var rows []row
	err := s.db.Raw(`
		SELECT
			p.id AS problem_id,
			p.slug,
			COUNT(s.id) AS attempts,
			BOOL_OR(s.success) AS has_success,
			COALESCE(MIN(CASE WHEN s.success THEN s.total_time END), 0) AS best_time
		FROM submissions s
		JOIN problems p ON p.id = s.problem_id
		WHERE s.user_id = ?
		GROUP BY p.id, p.slug
		ORDER BY p.slug
	`, userID).Scan(&rows).Error
	if err != nil {
		return nil, fmt.Errorf("failed to query progress: %w", err)
	}

	entries := make([]ProgressEntry, len(rows))
	for i, r := range rows {
		entries[i] = ProgressEntry{
			ProblemID: r.ProblemID,
			Slug:      r.Slug,
			Solved:    r.HasSuccess,
			Attempts:  r.Attempts,
			BestTime:  r.BestTime,
		}
	}

	return entries, nil
}

func (s *SubmissionService) GetSubmissions(userID uuid.UUID, page, limit int) ([]model.Submission, int64, error) {
	if page < 1 {
		page = 1
	}
	if limit < 1 || limit > 100 {
		limit = 20
	}

	var total int64
	if err := s.db.Model(&model.Submission{}).Where("user_id = ?", userID).Count(&total).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to count submissions: %w", err)
	}

	var submissions []model.Submission
	offset := (page - 1) * limit
	if err := s.db.Where("user_id = ?", userID).
		Preload("Problem").
		Order("created_at DESC").
		Offset(offset).Limit(limit).
		Find(&submissions).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to query submissions: %w", err)
	}

	return submissions, total, nil
}

func difficultyBonus(difficulty string) int {
	switch difficulty {
	case "Easy":
		return 5
	case "Medium":
		return 10
	case "Hard":
		return 20
	default:
		return 0
	}
}
