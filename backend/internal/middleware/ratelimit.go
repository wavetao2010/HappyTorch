package middleware

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// RateLimitByIP returns middleware that rate-limits requests by client IP.
func RateLimitByIP(rdb *redis.Client, limit int, window time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		key := fmt.Sprintf("rl:ip:%d:%s", limit, c.ClientIP())
		if blocked := checkRateLimit(c, rdb, key, limit, window); blocked {
			return
		}
		c.Next()
	}
}

// RateLimitByUser returns middleware that rate-limits requests by authenticated user ID.
// Must be placed after AuthRequired middleware.
func RateLimitByUser(rdb *redis.Client, limit int, window time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		userID, exists := c.Get(ContextUserID)
		if !exists {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "authentication required"})
			return
		}
		key := fmt.Sprintf("rl:user:%s", userID.(uuid.UUID).String())
		if blocked := checkRateLimit(c, rdb, key, limit, window); blocked {
			return
		}
		c.Next()
	}
}

// checkRateLimit implements a sliding window counter using Redis.
// Returns true if the request is rate-limited (and the response has been written).
func checkRateLimit(c *gin.Context, rdb *redis.Client, key string, limit int, window time.Duration) bool {
	ctx := context.Background()
	now := time.Now().UnixMilli()
	windowStart := now - window.Milliseconds()

	pipe := rdb.Pipeline()
	// Remove entries outside the window
	pipe.ZRemRangeByScore(ctx, key, "0", strconv.FormatInt(windowStart, 10))
	// Count entries in the window
	countCmd := pipe.ZCard(ctx, key)
	// Add the current request
	pipe.ZAdd(ctx, key, redis.Z{Score: float64(now), Member: now})
	// Set expiry on the key
	pipe.Expire(ctx, key, window)

	if _, err := pipe.Exec(ctx); err != nil {
		// On Redis error, allow the request through
		return false
	}

	count := countCmd.Val()
	if count >= int64(limit) {
		retryAfter := int(window.Seconds())
		c.Header("Retry-After", strconv.Itoa(retryAfter))
		c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
			"error": "rate limit exceeded",
		})
		return true
	}

	return false
}
