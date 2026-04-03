package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/happytorch/backend/internal/middleware"
	"github.com/happytorch/backend/internal/service"
)

type AuthHandler struct {
	authService        *service.AuthService
	jwtSecret          string
	gitHubClientID     string
	gitHubClientSecret string
	frontURL           string
}

func NewAuthHandler(authService *service.AuthService, jwtSecret, gitHubClientID, gitHubClientSecret, frontURL string) *AuthHandler {
	return &AuthHandler{
		authService:        authService,
		jwtSecret:          jwtSecret,
		gitHubClientID:     gitHubClientID,
		gitHubClientSecret: gitHubClientSecret,
		frontURL:           frontURL,
	}
}

func (h *AuthHandler) Register(c *gin.Context) {
	var input service.RegisterInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	user, err := h.authService.Register(input)
	if err != nil {
		c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
		return
	}

	accessToken, err := middleware.GenerateAccessToken(user.ID, h.jwtSecret)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to generate token"})
		return
	}

	refreshToken, err := middleware.GenerateRefreshToken(user.ID, h.jwtSecret)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to generate token"})
		return
	}

	middleware.SetTokenCookies(c, accessToken, refreshToken)
	c.JSON(http.StatusCreated, gin.H{"user": user})
}

func (h *AuthHandler) Login(c *gin.Context) {
	var input service.LoginInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	user, err := h.authService.Login(input)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}

	accessToken, err := middleware.GenerateAccessToken(user.ID, h.jwtSecret)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to generate token"})
		return
	}

	refreshToken, err := middleware.GenerateRefreshToken(user.ID, h.jwtSecret)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to generate token"})
		return
	}

	middleware.SetTokenCookies(c, accessToken, refreshToken)
	c.JSON(http.StatusOK, gin.H{"user": user})
}

func (h *AuthHandler) Logout(c *gin.Context) {
	middleware.ClearTokenCookies(c)
	c.JSON(http.StatusOK, gin.H{"message": "logged out"})
}

func (h *AuthHandler) Me(c *gin.Context) {
	userID, exists := c.Get(middleware.ContextUserID)
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "not authenticated"})
		return
	}

	user, err := h.authService.GetUserByID(userID.(uuid.UUID))
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "user not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"user": user})
}

// GitHubLogin redirects to GitHub OAuth authorization page
func (h *AuthHandler) GitHubLogin(c *gin.Context) {
	if h.gitHubClientID == "" {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "GitHub OAuth not configured"})
		return
	}

	redirectURI := c.Query("redirect")
	if redirectURI == "" {
		redirectURI = "/problems"
	}

	// Store redirect in a cookie for callback
	c.SetCookie("oauth_redirect", redirectURI, 600, "/", "", false, true)

	authURL := "https://github.com/login/oauth/authorize?client_id=" + h.gitHubClientID + "&scope=user:email"
	c.Redirect(http.StatusTemporaryRedirect, authURL)
}

// GitHubCallback handles the OAuth callback from GitHub
func (h *AuthHandler) GitHubCallback(c *gin.Context) {
	code := c.Query("code")
	if code == "" {
		c.Redirect(http.StatusTemporaryRedirect, h.frontURL+"/login?error=no_code")
		return
	}

	// Exchange code for access token
	accessToken, err := h.authService.ExchangeGitHubCode(code, h.gitHubClientID, h.gitHubClientSecret)
	if err != nil {
		c.Redirect(http.StatusTemporaryRedirect, h.frontURL+"/login?error=exchange_failed")
		return
	}

	// Get GitHub user info
	ghUser, err := h.authService.GetGitHubUser(accessToken)
	if err != nil {
		c.Redirect(http.StatusTemporaryRedirect, h.frontURL+"/login?error=user_fetch_failed")
		return
	}

	// Find or create user
	user, err := h.authService.FindOrCreateGitHubUser(ghUser)
	if err != nil {
		c.Redirect(http.StatusTemporaryRedirect, h.frontURL+"/login?error=create_failed")
		return
	}

	// Generate JWT tokens
	jwtAccessToken, err := middleware.GenerateAccessToken(user.ID, h.jwtSecret)
	if err != nil {
		c.Redirect(http.StatusTemporaryRedirect, h.frontURL+"/login?error=token_failed")
		return
	}

	refreshToken, err := middleware.GenerateRefreshToken(user.ID, h.jwtSecret)
	if err != nil {
		c.Redirect(http.StatusTemporaryRedirect, h.frontURL+"/login?error=token_failed")
		return
	}

	// Set auth cookies
	middleware.SetTokenCookies(c, jwtAccessToken, refreshToken)

	// Get redirect path from cookie
	redirectPath, _ := c.Cookie("oauth_redirect")
	if redirectPath == "" {
		redirectPath = "/problems"
	}
	c.SetCookie("oauth_redirect", "", -1, "/", "", false, true) // Clear the cookie

	c.Redirect(http.StatusTemporaryRedirect, h.frontURL+redirectPath)
}
