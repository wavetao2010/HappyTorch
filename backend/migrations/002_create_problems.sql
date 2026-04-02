CREATE TABLE problems (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug              VARCHAR(64)  NOT NULL UNIQUE,
    title             TEXT         NOT NULL,
    difficulty        VARCHAR(16)  NOT NULL,
    category          VARCHAR(64)  NOT NULL,
    function_name     VARCHAR(64),
    description       TEXT,
    signature         TEXT,
    example           TEXT,
    hint              TEXT,
    template_code     TEXT,
    tests_json        JSONB,
    solution_code     TEXT,
    solution_markdown TEXT,
    status            VARCHAR(16)  NOT NULL DEFAULT 'approved',
    author_id         UUID         REFERENCES users(id),
    reviewer_id       UUID         REFERENCES users(id),
    review_note       TEXT,
    sort_order        INTEGER      NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_problems_category   ON problems(category);
CREATE INDEX idx_problems_difficulty ON problems(difficulty);
CREATE INDEX idx_problems_status     ON problems(status);
