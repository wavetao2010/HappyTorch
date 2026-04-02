package service

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

const leaderboardTTL = 5 * time.Minute

// CacheService provides Redis-backed caching for expensive queries.
type CacheService struct {
	rdb *redis.Client
}

func NewCacheService(rdb *redis.Client) *CacheService {
	return &CacheService{rdb: rdb}
}

func leaderboardKey(sortBy string, limit int) string {
	return fmt.Sprintf("leaderboard:%s:%d", sortBy, limit)
}

// GetLeaderboard retrieves a cached leaderboard result. Returns nil on cache miss.
func (s *CacheService) GetLeaderboard(sortBy string, limit int) ([]LeaderboardEntry, error) {
	ctx := context.Background()
	data, err := s.rdb.Get(ctx, leaderboardKey(sortBy, limit)).Bytes()
	if err != nil {
		return nil, err
	}

	var entries []LeaderboardEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("failed to unmarshal cached leaderboard: %w", err)
	}
	return entries, nil
}

// SetLeaderboard caches a leaderboard result.
func (s *CacheService) SetLeaderboard(sortBy string, limit int, entries []LeaderboardEntry) error {
	ctx := context.Background()
	data, err := json.Marshal(entries)
	if err != nil {
		return fmt.Errorf("failed to marshal leaderboard for cache: %w", err)
	}
	return s.rdb.Set(ctx, leaderboardKey(sortBy, limit), data, leaderboardTTL).Err()
}

// InvalidateLeaderboard removes all cached leaderboard entries.
func (s *CacheService) InvalidateLeaderboard() {
	ctx := context.Background()
	iter := s.rdb.Scan(ctx, 0, "leaderboard:*", 100).Iterator()
	for iter.Next(ctx) {
		s.rdb.Del(ctx, iter.Val())
	}
}
