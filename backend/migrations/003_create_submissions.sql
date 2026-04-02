CREATE TABLE submissions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID         NOT NULL REFERENCES users(id),
    problem_id   UUID         NOT NULL REFERENCES problems(id),
    code         TEXT         NOT NULL,
    passed       INTEGER      NOT NULL DEFAULT 0,
    total        INTEGER      NOT NULL DEFAULT 0,
    total_time   DOUBLE PRECISION NOT NULL DEFAULT 0,
    success      BOOLEAN      NOT NULL DEFAULT false,
    results_json JSONB,
    output       TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_submissions_user_id    ON submissions(user_id);
CREATE INDEX idx_submissions_problem_id ON submissions(problem_id);
