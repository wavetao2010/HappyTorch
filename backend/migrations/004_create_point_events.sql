CREATE TABLE point_events (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID        NOT NULL REFERENCES users(id),
    event_type   VARCHAR(32) NOT NULL,
    points       INTEGER     NOT NULL,
    reference_id UUID,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_point_events_user_id ON point_events(user_id);
