CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        VARCHAR(32)  NOT NULL UNIQUE,
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   TEXT         NOT NULL,
    display_name    VARCHAR(64),
    avatar_url      TEXT,
    role            VARCHAR(16)  NOT NULL DEFAULT 'user',
    points          INTEGER      NOT NULL DEFAULT 0,
    problems_solved INTEGER      NOT NULL DEFAULT 0,
    bio             TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
