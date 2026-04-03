package main

import (
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"

	"github.com/happytorch/backend/internal/config"
	"github.com/happytorch/backend/internal/handler"
	"github.com/happytorch/backend/internal/judge"
	"github.com/happytorch/backend/internal/llm"
	"github.com/happytorch/backend/internal/middleware"
	"github.com/happytorch/backend/internal/model"
	"github.com/happytorch/backend/internal/service"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	db, err := gorm.Open(postgres.Open(cfg.DSN()), &gorm.Config{})
	if err != nil {
		log.Fatalf("failed to connect to database: %v", err)
	}

	// Auto-migrate database tables
	if err := db.AutoMigrate(
		&model.User{},
		&model.Problem{},
		&model.Submission{},
		&model.PointEvent{},
		&model.RoleApplication{},
	); err != nil {
		log.Fatalf("failed to auto-migrate database: %v", err)
	}
	log.Println("database migration completed")

	rdb := redis.NewClient(&redis.Options{
		Addr: cfg.RedisURL,
	})

	cacheService := service.NewCacheService(rdb)
	authService := service.NewAuthService(db)
	problemService := service.NewProblemService(db)
	judgeClient := judge.NewClient(cfg.JudgeURL)
	pointService := service.NewPointService(db)
	submissionService := service.NewSubmissionService(db, judgeClient, pointService, cacheService)
	communityService := service.NewCommunityService(db, judgeClient, pointService, cacheService)
	communityService.SetAutoApprove(cfg.AIAutoApprove)

	// Configure LLM provider for AI review
	if cfg.LLMProvider != "none" && cfg.LLMAPIKey != "" {
		llmProvider := llm.NewProvider(llm.Config{
			Provider: cfg.LLMProvider,
			APIKey:   cfg.LLMAPIKey,
			Model:    cfg.LLMModel,
		})
		communityService.GetAIReviewService().SetLLMProvider(llmProvider)
		log.Printf("LLM review enabled: %s", cfg.LLMProvider)
	}

	authHandler := handler.NewAuthHandler(authService, cfg.JWTSecret, cfg.GitHubClientID, cfg.GitHubClientSecret, cfg.FrontURL)
	problemHandler := handler.NewProblemHandler(problemService)
	submissionHandler := handler.NewSubmissionHandler(submissionService, problemService)
	progressHandler := handler.NewProgressHandler(db, submissionService)
	communityHandler := handler.NewCommunityHandler(communityService)

	r := gin.Default()
	r.Use(middleware.CORS(cfg.FrontURL))

	api := r.Group("/api")
	api.Use(middleware.RateLimitByIP(rdb, 100, time.Minute))
	{
		api.GET("/health", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"status": "ok"})
		})
		auth := api.Group("/auth")
		{
			auth.POST("/register", middleware.RateLimitByIP(rdb, 5, time.Minute), authHandler.Register)
			auth.POST("/login", middleware.RateLimitByIP(rdb, 5, time.Minute), authHandler.Login)
			auth.POST("/logout", authHandler.Logout)
			auth.GET("/me", middleware.AuthRequired(cfg.JWTSecret), authHandler.Me)
			auth.GET("/github", authHandler.GitHubLogin)
			auth.GET("/github/callback", authHandler.GitHubCallback)
		}

		problems := api.Group("/problems")
		{
			problems.GET("", problemHandler.List)
			problems.GET("/:slug", problemHandler.GetBySlug)
			problems.POST("/:slug/submit",
				middleware.AuthRequired(cfg.JWTSecret),
				middleware.RateLimitByUser(rdb, 10, time.Minute),
				submissionHandler.Submit,
			)
			problems.GET("/:slug/solution", middleware.AuthRequired(cfg.JWTSecret), submissionHandler.GetSolution)
			problems.POST("/submit", middleware.AuthRequired(cfg.JWTSecret), communityHandler.SubmitProblem)
			problems.GET("/mine", middleware.AuthRequired(cfg.JWTSecret), communityHandler.ListMyProblems)
			problems.PUT("/:id", middleware.AuthRequired(cfg.JWTSecret), communityHandler.UpdateMyProblem)
		}

		users := api.Group("/users")
		{
			users.GET("/:username", communityHandler.GetUserProfile)

			me := users.Group("", middleware.AuthRequired(cfg.JWTSecret))
			{
				me.GET("/me/progress", progressHandler.GetProgress)
				me.GET("/me/submissions", progressHandler.GetSubmissions)
			}
		}

		api.GET("/leaderboard", communityHandler.GetLeaderboard)

		// Role applications
		applications := api.Group("/applications", middleware.AuthRequired(cfg.JWTSecret))
		{
			applications.POST("/moderator", communityHandler.ApplyForModerator)
			applications.GET("/mine", communityHandler.GetMyApplications)
		}

		admin := api.Group("/admin", middleware.AuthRequired(cfg.JWTSecret))
		{
			adminProblems := admin.Group("/problems", middleware.RoleRequired(db, "moderator"))
			{
				adminProblems.GET("/pending", communityHandler.ListPendingProblems)
				adminProblems.GET("/:id", communityHandler.GetPendingProblem)
				adminProblems.POST("/:id/review", communityHandler.ReviewProblem)
			}

			adminUsers := admin.Group("/users", middleware.RoleRequired(db, "admin"))
			{
				adminUsers.GET("", communityHandler.ListUsers)
				adminUsers.PUT("/:id/role", communityHandler.ChangeUserRole)
			}

			adminApps := admin.Group("/applications", middleware.RoleRequired(db, "admin"))
			{
				adminApps.GET("", communityHandler.ListRoleApplications)
				adminApps.POST("/:id/review", communityHandler.ReviewApplication)
			}
		}
	}

	log.Printf("starting server on :%s", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatalf("failed to start server: %v", err)
	}
}
