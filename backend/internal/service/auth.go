package service

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"

	"github.com/happytorch/backend/internal/model"
)

type AuthService struct {
	db *gorm.DB
}

func NewAuthService(db *gorm.DB) *AuthService {
	return &AuthService{db: db}
}

type RegisterInput struct {
	Username string `json:"username" binding:"required,min=3,max=32"`
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required,min=6"`
}

type LoginInput struct {
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required"`
}

func (s *AuthService) Register(input RegisterInput) (*model.User, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(input.Password), bcrypt.DefaultCost)
	if err != nil {
		return nil, fmt.Errorf("failed to hash password: %w", err)
	}

	// First user becomes admin
	role := "user"
	var count int64
	s.db.Model(&model.User{}).Count(&count)
	if count == 0 {
		role = "admin"
	}

	user := &model.User{
		ID:           uuid.New(),
		Username:     input.Username,
		Email:        input.Email,
		PasswordHash: string(hash),
		DisplayName:  input.Username,
		Role:         role,
	}

	if err := s.db.Create(user).Error; err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	return user, nil
}

func (s *AuthService) Login(input LoginInput) (*model.User, error) {
	var user model.User
	if err := s.db.Where("email = ?", input.Email).First(&user).Error; err != nil {
		return nil, fmt.Errorf("invalid credentials")
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(input.Password)); err != nil {
		return nil, fmt.Errorf("invalid credentials")
	}

	return &user, nil
}

func (s *AuthService) GetUserByID(id uuid.UUID) (*model.User, error) {
	var user model.User
	if err := s.db.First(&user, "id = ?", id).Error; err != nil {
		return nil, fmt.Errorf("failed to find user: %w", err)
	}
	return &user, nil
}

// GitHubUser represents the user info from GitHub API
type GitHubUser struct {
	ID        int64  `json:"id"`
	Login     string `json:"login"`
	Email     string `json:"email"`
	Name      string `json:"name"`
	AvatarURL string `json:"avatar_url"`
}

// ExchangeGitHubCode exchanges the OAuth code for an access token
func (s *AuthService) ExchangeGitHubCode(code, clientID, clientSecret string) (string, error) {
	data := url.Values{}
	data.Set("client_id", clientID)
	data.Set("client_secret", clientSecret)
	data.Set("code", code)

	req, err := http.NewRequest("POST", "https://github.com/login/oauth/access_token", strings.NewReader(data.Encode()))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to exchange code: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	var result struct {
		AccessToken string `json:"access_token"`
		Error       string `json:"error"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if result.Error != "" {
		return "", fmt.Errorf("github error: %s", result.Error)
	}

	return result.AccessToken, nil
}

// GetGitHubUser fetches user info from GitHub API
func (s *AuthService) GetGitHubUser(accessToken string) (*GitHubUser, error) {
	req, err := http.NewRequest("GET", "https://api.github.com/user", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Accept", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var ghUser GitHubUser
	if err := json.Unmarshal(body, &ghUser); err != nil {
		return nil, fmt.Errorf("failed to parse user: %w", err)
	}

	return &ghUser, nil
}

// FindOrCreateGitHubUser finds existing user by GitHub ID or creates a new one
func (s *AuthService) FindOrCreateGitHubUser(ghUser *GitHubUser) (*model.User, error) {
	var user model.User

	// Try to find by GitHub ID
	err := s.db.Where("github_id = ?", ghUser.ID).First(&user).Error
	if err == nil {
		// Update avatar if changed
		if user.AvatarURL != ghUser.AvatarURL {
			s.db.Model(&user).Update("avatar_url", ghUser.AvatarURL)
			user.AvatarURL = ghUser.AvatarURL
		}
		return &user, nil
	}

	// Try to find by email (link existing account)
	if ghUser.Email != "" {
		err = s.db.Where("email = ?", ghUser.Email).First(&user).Error
		if err == nil {
			// Link GitHub ID to existing account
			s.db.Model(&user).Updates(map[string]interface{}{
				"github_id":  ghUser.ID,
				"avatar_url": ghUser.AvatarURL,
			})
			user.GitHubID = ghUser.ID
			user.AvatarURL = ghUser.AvatarURL
			return &user, nil
		}
	}

	// Create new user
	displayName := ghUser.Name
	if displayName == "" {
		displayName = ghUser.Login
	}

	email := ghUser.Email
	if email == "" {
		email = fmt.Sprintf("%d+%s@users.noreply.github.com", ghUser.ID, ghUser.Login)
	}

	// First user becomes admin
	role := "user"
	var count int64
	s.db.Model(&model.User{}).Count(&count)
	if count == 0 {
		role = "admin"
	}

	user = model.User{
		ID:          uuid.New(),
		Username:    ghUser.Login,
		Email:       email,
		GitHubID:    ghUser.ID,
		DisplayName: displayName,
		AvatarURL:   ghUser.AvatarURL,
		Role:        role,
	}

	if err := s.db.Create(&user).Error; err != nil {
		// Username might conflict, try with suffix
		user.Username = fmt.Sprintf("%s_%d", ghUser.Login, ghUser.ID%10000)
		if err := s.db.Create(&user).Error; err != nil {
			return nil, fmt.Errorf("failed to create user: %w", err)
		}
	}

	return &user, nil
}
