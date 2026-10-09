-- AI plan, stage B (docs/AI_PLAN.md): recipe ideas for futures model search, and the cost
-- of every call to Claude.

-- Every recipe model search has tried or will try. by: who put it together, "haiku"
-- (Claude Haiku, queued until a search takes it) or "random" (a random mix, recorded
-- once tried, so Haiku can learn from it). result: TRAINING numbers only (best score,
-- gates, days traded, profit at normal and double costs); held-out and lockbox numbers
-- never go here, because this table is what Haiku is shown.
CREATE TABLE recipe_ideas (
  id          bigserial PRIMARY KEY,
  family      text NOT NULL,                -- recipe_... (the name of the recipe)
  recipe      jsonb NOT NULL,
  by          text NOT NULL CHECK (by IN ('haiku', 'random')),
  note        text,                         -- Haiku's one sentence on the idea behind it
  status      text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'taken', 'tried')),
  job_id      uuid REFERENCES jobs(id),     -- the search that took it
  result      jsonb,
  tries       integer NOT NULL DEFAULT 0,   -- settings tried with this recipe so far
  created_at  timestamptz NOT NULL DEFAULT now(),
  taken_at    timestamptz,
  tried_at    timestamptz
);
CREATE INDEX recipe_ideas_queue ON recipe_ideas (status, id);
CREATE UNIQUE INDEX recipe_ideas_one_random ON recipe_ideas (family) WHERE by = 'random';

-- Every call to Claude, with what it cost (the monthly cap in config/ai.toml is checked
-- against this before each call).
CREATE TABLE ai_calls (
  id             bigserial PRIMARY KEY,
  at             timestamptz NOT NULL DEFAULT now(),
  feature        text NOT NULL,             -- "recipes" (stage B)
  model          text NOT NULL,
  input_tokens   integer NOT NULL DEFAULT 0,
  output_tokens  integer NOT NULL DEFAULT 0,
  cost_usd       double precision NOT NULL DEFAULT 0,
  outcome        text NOT NULL,             -- "ok", "refused", "error: ...", "bad: ..."
  detail         jsonb
);
CREATE INDEX ai_calls_at ON ai_calls (at);
