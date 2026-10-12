-- AI plan, stages C and D (docs/AI_PLAN.md): Claude Haiku's reviews of futures models,
-- and its daily market note. Advice only: nothing here changes a model or places an
-- order, and neither is ever shown to the recipe writer.

-- Reviews of a model's numbers. key: what the review was of (the backtest time, the
-- Final check and the paper days so far), so a model is reviewed again only when its
-- results change. status: requested (the owner asked, or it is due), done or failed.
CREATE TABLE model_reviews (
  id            bigserial PRIMARY KEY,
  model_id      text NOT NULL REFERENCES models(id),
  key           text NOT NULL,
  status        text NOT NULL DEFAULT 'requested' CHECK (status IN ('requested', 'done', 'failed')),
  asked_by      text NOT NULL CHECK (asked_by IN ('owner', 'auto')),
  review        jsonb,
  error         text,
  requested_at  timestamptz NOT NULL DEFAULT now(),
  done_at       timestamptz
);
CREATE INDEX model_reviews_latest ON model_reviews (model_id, id DESC);
CREATE INDEX model_reviews_waiting ON model_reviews (status, id) WHERE status = 'requested';

-- One market note per trading day (Chicago date), written from the night's and the
-- morning's headlines. before_open: written before the session opened, so it can only
-- have used news known in advance (only those count when paper results on flagged days
-- are compared with the rest).
CREATE TABLE market_notes (
  day          date PRIMARY KEY,
  written_at   timestamptz NOT NULL DEFAULT now(),
  before_open  boolean NOT NULL,
  headlines    integer NOT NULL DEFAULT 0,
  flagged      boolean NOT NULL DEFAULT false,
  note         jsonb NOT NULL
);
