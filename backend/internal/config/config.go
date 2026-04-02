package config

import (
	"fmt"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	DBHost     string
	DBPort     string
	DBUser     string
	DBPassword string
	DBName     string
	DBSSLMode  string

	JWTSecret          string
	JudgeURL           string
	Port               string
	FrontURL           string
	RedisURL           string
	GitHubClientID     string
	GitHubClientSecret string
	AIAutoApprove      bool
	LLMProvider        string // gemini, qwen, claude, openai, none
	LLMAPIKey          string
	LLMModel           string // optional model override
}

func Load() (*Config, error) {
	_ = godotenv.Load()

	cfg := &Config{
		DBHost:             getEnv("DB_HOST", "localhost"),
		DBPort:             getEnv("DB_PORT", "5432"),
		DBUser:             getEnv("DB_USER", "postgres"),
		DBPassword:         getEnv("DB_PASSWORD", "postgres"),
		DBName:             getEnv("DB_NAME", "happytorch"),
		DBSSLMode:          getEnv("DB_SSLMODE", "disable"),
		JWTSecret:          os.Getenv("JWT_SECRET"),
		JudgeURL:           getEnv("JUDGE_URL", "http://localhost:9000"),
		Port:               getEnv("PORT", "8080"),
		FrontURL:           getEnv("FRONT_URL", "http://localhost:3000"),
		RedisURL:           getEnv("REDIS_URL", "localhost:6379"),
		GitHubClientID:     os.Getenv("GITHUB_CLIENT_ID"),
		GitHubClientSecret: os.Getenv("GITHUB_CLIENT_SECRET"),
		AIAutoApprove:      os.Getenv("AI_AUTO_APPROVE") == "true",
		LLMProvider:        getEnv("LLM_PROVIDER", "none"),
		LLMAPIKey:          os.Getenv("LLM_API_KEY"),
		LLMModel:           os.Getenv("LLM_MODEL"),
	}

	if cfg.JWTSecret == "" {
		return nil, fmt.Errorf("JWT_SECRET environment variable is required")
	}

	return cfg, nil
}

func (c *Config) DSN() string {
	return fmt.Sprintf(
		"host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
		c.DBHost, c.DBPort, c.DBUser, c.DBPassword, c.DBName, c.DBSSLMode,
	)
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
