package service

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/gosimple/slug"
	"gorm.io/gorm"

	"github.com/happytorch/backend/internal/judge"
	"github.com/happytorch/backend/internal/model"
)

type CommunityService struct {
	db              *gorm.DB
	judgeClient     *judge.Client
	pointService    *PointService
	cacheService    *CacheService
	aiReviewService *AIReviewService
	autoApprove     bool // If true, AI-approved problems go live immediately
}

func NewCommunityService(db *gorm.DB, judgeClient *judge.Client, pointService *PointService, cacheService *CacheService) *CommunityService {
	return &CommunityService{
		db:              db,
		judgeClient:     judgeClient,
		pointService:    pointService,
		cacheService:    cacheService,
		aiReviewService: NewAIReviewService(judgeClient),
		autoApprove:     false, // Default: require human review after AI approval
	}
}

// SetAutoApprove configures whether AI-approved problems go live immediately
func (s *CommunityService) SetAutoApprove(enabled bool) {
	s.autoApprove = enabled
}

// GetAIReviewService returns the AI review service for LLM configuration
func (s *CommunityService) GetAIReviewService() *AIReviewService {
	return s.aiReviewService
}

// --- Problem submission ---

type TestInput struct {
	Name string `json:"name" binding:"required"`
	Code string `json:"code" binding:"required"`
}

type SubmitProblemInput struct {
	Title            string     `json:"title" binding:"required"`
	Difficulty       string     `json:"difficulty" binding:"required,oneof=Easy Medium Hard"`
	Category         string     `json:"category" binding:"required"`
	FunctionName     string     `json:"function_name" binding:"required"`
	Description      string     `json:"description" binding:"required"`
	Hint             string     `json:"hint"`
	TemplateCode     string     `json:"template_code" binding:"required"`
	Tests            []TestInput `json:"tests" binding:"required,min=1"`
	SolutionCode     string     `json:"solution_code"`
	SolutionMarkdown string     `json:"solution_markdown"`
}

// ProblemSubmitResult contains the problem and review result
type ProblemSubmitResult struct {
	Problem *model.Problem `json:"problem"`
	Review  *ReviewResult  `json:"review"`
}

func (s *CommunityService) SubmitProblem(authorID uuid.UUID, input SubmitProblemInput) (*ProblemSubmitResult, error) {
	// Basic validation
	if !strings.Contains(input.TemplateCode, input.FunctionName) {
		return nil, fmt.Errorf("template_code must contain function_name %q", input.FunctionName)
	}

	for i, t := range input.Tests {
		if !strings.Contains(t.Code, "{fn}") {
			return nil, fmt.Errorf("test %d (%s) must contain {fn} placeholder", i, t.Name)
		}
	}

	// Run AI review
	review, err := s.aiReviewService.ReviewProblem(input)
	if err != nil {
		return nil, fmt.Errorf("AI review failed: %w", err)
	}

	// If AI review found critical issues, reject immediately
	if !review.Passed {
		return &ProblemSubmitResult{
			Problem: nil,
			Review:  review,
		}, nil
	}

	testsJSON, err := json.Marshal(input.Tests)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tests: %w", err)
	}

	problemSlug := slug.Make(input.Title)
	// Ensure uniqueness
	var count int64
	s.db.Model(&model.Problem{}).Where("slug = ?", problemSlug).Count(&count)
	if count > 0 {
		problemSlug = fmt.Sprintf("%s-%s", problemSlug, uuid.New().String()[:8])
	}

	// Determine status based on AI review and auto-approve setting
	status := "ai_approved"
	if s.autoApprove && review.Score >= 80 {
		status = "approved"
	}

	problem := &model.Problem{
		Slug:             problemSlug,
		Title:            input.Title,
		Difficulty:       input.Difficulty,
		Category:         input.Category,
		FunctionName:     input.FunctionName,
		Description:      input.Description,
		Hint:             input.Hint,
		TemplateCode:     input.TemplateCode,
		TestsJSON:        testsJSON,
		SolutionCode:     input.SolutionCode,
		SolutionMarkdown: input.SolutionMarkdown,
		Status:           status,
		AuthorID:         &authorID,
		AIScore:          review.Score,
	}

	if err := s.db.Create(problem).Error; err != nil {
		return nil, fmt.Errorf("failed to create problem: %w", err)
	}

	// If auto-approved, award points
	if status == "approved" {
		_ = s.pointService.AwardPoints(authorID, "problem_approved", 50, problem.ID)
		_ = s.db.Model(&model.User{}).Where("id = ?", authorID).
			Update("points", gorm.Expr("points + ?", 50)).Error
		if s.cacheService != nil {
			s.cacheService.InvalidateLeaderboard()
		}
	}

	return &ProblemSubmitResult{
		Problem: problem,
		Review:  review,
	}, nil
}

func (s *CommunityService) ListMyProblems(authorID uuid.UUID) ([]model.Problem, error) {
	var problems []model.Problem
	if err := s.db.Where("author_id = ?", authorID).Order("created_at DESC").Find(&problems).Error; err != nil {
		return nil, fmt.Errorf("failed to list user problems: %w", err)
	}
	return problems, nil
}

func (s *CommunityService) UpdateMyProblem(authorID uuid.UUID, problemID uuid.UUID, input SubmitProblemInput) (*model.Problem, error) {
	var problem model.Problem
	if err := s.db.First(&problem, "id = ?", problemID).Error; err != nil {
		return nil, fmt.Errorf("failed to find problem: %w", err)
	}

	if problem.AuthorID == nil || *problem.AuthorID != authorID {
		return nil, fmt.Errorf("you can only edit your own problems")
	}

	if problem.Status != "pending" && problem.Status != "rejected" {
		return nil, fmt.Errorf("can only edit problems with status pending or rejected")
	}

	if !strings.Contains(input.TemplateCode, input.FunctionName) {
		return nil, fmt.Errorf("template_code must contain function_name %q", input.FunctionName)
	}

	for i, t := range input.Tests {
		if !strings.Contains(t.Code, "{fn}") {
			return nil, fmt.Errorf("test %d (%s) must contain {fn} placeholder", i, t.Name)
		}
	}

	testsJSON, err := json.Marshal(input.Tests)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tests: %w", err)
	}

	problem.Title = input.Title
	problem.Difficulty = input.Difficulty
	problem.Category = input.Category
	problem.FunctionName = input.FunctionName
	problem.Description = input.Description
	problem.Hint = input.Hint
	problem.TemplateCode = input.TemplateCode
	problem.TestsJSON = testsJSON
	problem.SolutionCode = input.SolutionCode
	problem.SolutionMarkdown = input.SolutionMarkdown
	problem.Status = "pending" // Reset to pending after edit

	if err := s.db.Save(&problem).Error; err != nil {
		return nil, fmt.Errorf("failed to update problem: %w", err)
	}

	return &problem, nil
}

// --- Admin / Moderator ---

func (s *CommunityService) ListPendingProblems() ([]model.Problem, error) {
	var problems []model.Problem
	// Include both "pending" (legacy) and "ai_approved" (new AI-reviewed)
	if err := s.db.Where("status IN ?", []string{"pending", "ai_approved"}).
		Preload("Author").
		Order("ai_score DESC, created_at ASC").
		Find(&problems).Error; err != nil {
		return nil, fmt.Errorf("failed to list pending problems: %w", err)
	}
	return problems, nil
}

func (s *CommunityService) GetPendingProblem(problemID uuid.UUID) (*model.Problem, error) {
	var problem model.Problem
	if err := s.db.Preload("Author").First(&problem, "id = ?", problemID).Error; err != nil {
		return nil, fmt.Errorf("failed to find problem: %w", err)
	}
	return &problem, nil
}

type ReviewInput struct {
	Action string `json:"action" binding:"required,oneof=approve reject"`
	Note   string `json:"note"`
}

func (s *CommunityService) ReviewProblem(reviewerID uuid.UUID, problemID uuid.UUID, input ReviewInput) (*model.Problem, error) {
	var problem model.Problem
	if err := s.db.First(&problem, "id = ?", problemID).Error; err != nil {
		return nil, fmt.Errorf("failed to find problem: %w", err)
	}

	if problem.Status != "pending" && problem.Status != "ai_approved" {
		return nil, fmt.Errorf("problem is not pending review")
	}

	problem.ReviewerID = &reviewerID
	problem.ReviewNote = input.Note

	switch input.Action {
	case "approve":
		problem.Status = "approved"
	case "reject":
		problem.Status = "rejected"
	}

	if err := s.db.Save(&problem).Error; err != nil {
		return nil, fmt.Errorf("failed to update problem review: %w", err)
	}

	// Award 50 points to author for approved community problem
	if input.Action == "approve" && problem.AuthorID != nil {
		_ = s.pointService.AwardPoints(*problem.AuthorID, "problem_approved", 50, problem.ID)
		_ = s.db.Model(&model.User{}).Where("id = ?", *problem.AuthorID).
			Update("points", gorm.Expr("points + ?", 50)).Error
		if s.cacheService != nil {
			s.cacheService.InvalidateLeaderboard()
		}
	}

	return &problem, nil
}

// --- Admin user management ---

func (s *CommunityService) ListUsers(page, limit int) ([]model.User, int64, error) {
	if page < 1 {
		page = 1
	}
	if limit < 1 || limit > 100 {
		limit = 20
	}

	var total int64
	if err := s.db.Model(&model.User{}).Count(&total).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to count users: %w", err)
	}

	var users []model.User
	offset := (page - 1) * limit
	if err := s.db.Order("created_at DESC").Offset(offset).Limit(limit).Find(&users).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to list users: %w", err)
	}

	return users, total, nil
}

type ChangeRoleInput struct {
	Role string `json:"role" binding:"required,oneof=user contributor moderator admin"`
}

func (s *CommunityService) ChangeUserRole(userID uuid.UUID, input ChangeRoleInput) (*model.User, error) {
	var user model.User
	if err := s.db.First(&user, "id = ?", userID).Error; err != nil {
		return nil, fmt.Errorf("failed to find user: %w", err)
	}

	user.Role = input.Role
	if err := s.db.Save(&user).Error; err != nil {
		return nil, fmt.Errorf("failed to update user role: %w", err)
	}

	return &user, nil
}

// --- Role Applications ---

type ApplyModeratorInput struct {
	Reason string `json:"reason" binding:"required,min=20"`
}

func (s *CommunityService) ApplyForModerator(userID uuid.UUID, input ApplyModeratorInput) (*model.RoleApplication, error) {
	canApply, reason := s.pointService.CanApplyForModerator(userID)
	if !canApply {
		return nil, fmt.Errorf(reason)
	}

	app := &model.RoleApplication{
		UserID: userID,
		Role:   "moderator",
		Status: "pending",
		Reason: input.Reason,
	}

	if err := s.db.Create(app).Error; err != nil {
		return nil, fmt.Errorf("failed to create application: %w", err)
	}

	return app, nil
}

func (s *CommunityService) ListRoleApplications() ([]model.RoleApplication, error) {
	var apps []model.RoleApplication
	if err := s.db.Where("status = ?", "pending").
		Preload("User").
		Order("created_at ASC").
		Find(&apps).Error; err != nil {
		return nil, fmt.Errorf("failed to list applications: %w", err)
	}
	return apps, nil
}

type ReviewApplicationInput struct {
	Action string `json:"action" binding:"required,oneof=approve reject"`
	Note   string `json:"note"`
}

func (s *CommunityService) ReviewApplication(reviewerID uuid.UUID, appID uuid.UUID, input ReviewApplicationInput) (*model.RoleApplication, error) {
	var app model.RoleApplication
	if err := s.db.Preload("User").First(&app, "id = ?", appID).Error; err != nil {
		return nil, fmt.Errorf("application not found: %w", err)
	}

	if app.Status != "pending" {
		return nil, fmt.Errorf("application already reviewed")
	}

	app.ReviewerID = &reviewerID
	app.ReviewNote = input.Note

	if input.Action == "approve" {
		app.Status = "approved"
		// Update user role
		if err := s.db.Model(&model.User{}).Where("id = ?", app.UserID).Update("role", app.Role).Error; err != nil {
			return nil, fmt.Errorf("failed to update user role: %w", err)
		}
	} else {
		app.Status = "rejected"
	}

	if err := s.db.Save(&app).Error; err != nil {
		return nil, fmt.Errorf("failed to update application: %w", err)
	}

	return &app, nil
}

func (s *CommunityService) GetMyApplications(userID uuid.UUID) ([]model.RoleApplication, error) {
	var apps []model.RoleApplication
	if err := s.db.Where("user_id = ?", userID).
		Order("created_at DESC").
		Find(&apps).Error; err != nil {
		return nil, fmt.Errorf("failed to get applications: %w", err)
	}
	return apps, nil
}

// --- Leaderboard ---

type LeaderboardEntry struct {
	Rank          int       `json:"rank"`
	UserID        uuid.UUID `json:"user_id"`
	Username      string    `json:"username"`
	DisplayName   string    `json:"display_name"`
	AvatarURL     string    `json:"avatar_url"`
	Points        int       `json:"points"`
	ProblemsSolved int      `json:"problems_solved"`
	Contributions int       `json:"contributions"`
}

func (s *CommunityService) GetLeaderboard(sortBy string, limit int) ([]LeaderboardEntry, error) {
	if limit < 1 || limit > 100 {
		limit = 50
	}

	// Try cache first
	if s.cacheService != nil {
		if cached, err := s.cacheService.GetLeaderboard(sortBy, limit); err == nil && cached != nil {
			return cached, nil
		}
	}

	orderClause := "u.points DESC"
	switch sortBy {
	case "solved":
		orderClause = "u.problems_solved DESC"
	case "contributions":
		orderClause = "contributions DESC"
	}

	var entries []LeaderboardEntry
	err := s.db.Raw(fmt.Sprintf(`
		SELECT
			u.id AS user_id,
			u.username,
			u.display_name,
			u.avatar_url,
			u.points,
			u.problems_solved,
			COALESCE(c.cnt, 0) AS contributions
		FROM users u
		LEFT JOIN (
			SELECT author_id, COUNT(*) AS cnt
			FROM problems
			WHERE status = 'approved' AND author_id IS NOT NULL
			GROUP BY author_id
		) c ON c.author_id = u.id
		ORDER BY %s
		LIMIT ?
	`, orderClause), limit).Scan(&entries).Error
	if err != nil {
		return nil, fmt.Errorf("failed to query leaderboard: %w", err)
	}

	for i := range entries {
		entries[i].Rank = i + 1
	}

	// Store in cache
	if s.cacheService != nil {
		_ = s.cacheService.SetLeaderboard(sortBy, limit, entries)
	}

	return entries, nil
}

// --- Public user profile ---

type UserProfile struct {
	UserID         uuid.UUID `json:"user_id"`
	Username       string    `json:"username"`
	DisplayName    string    `json:"display_name"`
	AvatarURL      string    `json:"avatar_url"`
	Bio            string    `json:"bio"`
	Points         int       `json:"points"`
	ProblemsSolved int       `json:"problems_solved"`
	Contributions  int       `json:"contributions"`
	CreatedAt      string    `json:"created_at"`
}

func (s *CommunityService) GetUserProfile(username string) (*UserProfile, error) {
	var user model.User
	if err := s.db.Where("username = ?", username).First(&user).Error; err != nil {
		return nil, fmt.Errorf("failed to find user: %w", err)
	}

	var contributions int64
	s.db.Model(&model.Problem{}).Where("author_id = ? AND status = ?", user.ID, "approved").Count(&contributions)

	return &UserProfile{
		UserID:         user.ID,
		Username:       user.Username,
		DisplayName:    user.DisplayName,
		AvatarURL:      user.AvatarURL,
		Bio:            user.Bio,
		Points:         user.Points,
		ProblemsSolved: user.ProblemsSolved,
		Contributions:  int(contributions),
		CreatedAt:      user.CreatedAt.Format("2006-01-02T15:04:05Z"),
	}, nil
}
