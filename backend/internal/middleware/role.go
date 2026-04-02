package middleware

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gorm.io/gorm"

	"github.com/happytorch/backend/internal/model"
)

var roleLevel = map[string]int{
	"user":      0,
	"moderator": 1,
	"admin":     2,
}

func RoleRequired(db *gorm.DB, minRole string) gin.HandlerFunc {
	minLevel := roleLevel[minRole]

	return func(c *gin.Context) {
		userID, exists := c.Get(ContextUserID)
		if !exists {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "authentication required"})
			return
		}

		var user model.User
		if err := db.First(&user, "id = ?", userID.(uuid.UUID)).Error; err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "user not found"})
			return
		}

		level, ok := roleLevel[user.Role]
		if !ok || level < minLevel {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "insufficient permissions"})
			return
		}

		c.Set("user_role", user.Role)
		c.Next()
	}
}
