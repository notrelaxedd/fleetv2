-- News research engine (docs/NEWS_PLAN.md): articles, Claude's analyses and
-- predictions with their scores, and the news strategies' positions.

-- The two news strategies are models rows too (books and orders point at models).
ALTER TABLE models DROP CONSTRAINT models_origin_check;
ALTER TABLE models ADD CONSTRAINT models_origin_check CHECK (origin IN ('starter', 'search', 'news'));

-- Articles from Alpaca's news feed that name at least one symbol we trade.
CREATE TABLE news (
  id              bigint PRIMARY KEY,           -- Alpaca's article id
  headline        text NOT NULL,
  summary         text NOT NULL DEFAULT '',
  content         text NOT NULL DEFAULT '',     -- plain text, HTML removed
  author          text NOT NULL DEFAULT '',
  source          text NOT NULL DEFAULT '',
  url             text NOT NULL DEFAULT '',
  symbols         text[] NOT NULL,              -- every symbol Alpaca tagged, as given
  ours            text[] NOT NULL,              -- the tagged symbols in our universe ("BTC/USD" form)
  created_at      timestamptz NOT NULL,         -- published
  updated_at      timestamptz,
  fetched_at      timestamptz NOT NULL DEFAULT now(),
  -- pending: waiting for the live analyst; backfill: stored for replay only
  analysis_status text NOT NULL DEFAULT 'pending'
                  CHECK (analysis_status IN ('pending', 'backfill', 'analysed', 'skipped', 'failed', 'over_budget')),
  analysis_note   text
);
CREATE INDEX news_created_idx ON news (created_at DESC);
CREATE INDEX news_pending_idx ON news (created_at) WHERE analysis_status = 'pending';

-- One call to Claude about one article.
CREATE TABLE news_analyses (
  id                 bigserial PRIMARY KEY,
  news_id            bigint NOT NULL REFERENCES news(id),
  source             text NOT NULL CHECK (source IN ('live', 'replay')),
  ai_call_id         bigint REFERENCES ai_calls(id),   -- the spend record (coordinator.haiku.record_call)
  requested_model    text NOT NULL,
  served_model       text NOT NULL,             -- differs when a refusal fallback answered
  prompt_version     text NOT NULL,
  input_tokens       integer NOT NULL DEFAULT 0,
  output_tokens      integer NOT NULL DEFAULT 0,
  cache_read_tokens  integer NOT NULL DEFAULT 0,
  cache_write_tokens integer NOT NULL DEFAULT 0,
  cost_usd           double precision NOT NULL DEFAULT 0,
  stop_reason        text,
  context            jsonb NOT NULL,            -- the prices and times Claude was shown
  raw                jsonb,                     -- Claude's answer as received
  error              text,
  created_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX news_analyses_day_idx ON news_analyses (created_at);
CREATE INDEX news_analyses_news_idx ON news_analyses (news_id);

-- One prediction about one symbol, and (once its horizon has passed) its score.
CREATE TABLE news_predictions (
  id                bigserial PRIMARY KEY,
  analysis_id       bigint NOT NULL REFERENCES news_analyses(id),
  news_id           bigint NOT NULL REFERENCES news(id),
  source            text NOT NULL CHECK (source IN ('live', 'replay')),
  model             text NOT NULL,              -- the model that answered (served_model)
  symbol            text NOT NULL,
  market            text NOT NULL CHECK (market IN ('stocks', 'crypto')),
  direction         text NOT NULL CHECK (direction IN ('up', 'down', 'none')),
  expected_move_pct double precision NOT NULL,  -- size of the move, always >= 0
  horizon_days      integer NOT NULL CHECK (horizon_days BETWEEN 1 AND 10),
  confidence        double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  already_priced_in boolean NOT NULL,
  rationale         text NOT NULL DEFAULT '',
  decided_at        timestamptz NOT NULL,       -- when the prediction counts from (replay: article time + delay)
  price_at_decision double precision,
  -- score (NULL until scored)
  entry_ts          timestamptz,                -- bar whose open is the entry
  exit_ts           timestamptz,                -- bar whose close is the exit
  entry_price       double precision,
  exit_price        double precision,
  move_pct          double precision,           -- the symbol's raw move, entry to exit
  bench_pct         double precision,           -- SPY or BTC/USD over the same bars
  signed_pct        double precision,           -- direction x move, after costs
  twin_direction    text CHECK (twin_direction IN ('up', 'down')),
  twin_signed_pct   double precision,           -- the coin-flip twin, after the same costs
  hit               boolean,                    -- direction x (move - bench) > 0
  scored_at         timestamptz,
  score_note        text                        -- why it cannot be scored, if so
);
CREATE INDEX news_predictions_news_idx ON news_predictions (news_id);
CREATE INDEX news_predictions_unscored_idx ON news_predictions (decided_at) WHERE scored_at IS NULL AND direction <> 'none';
CREATE INDEX news_predictions_market_idx ON news_predictions (market, source, scored_at);

-- A position a news strategy opened because of one prediction.
CREATE TABLE news_positions (
  id             bigserial PRIMARY KEY,
  book_id        bigint NOT NULL REFERENCES books(id),
  prediction_id  bigint NOT NULL REFERENCES news_predictions(id),
  symbol         text NOT NULL,
  side           text NOT NULL CHECK (side IN ('long', 'short')),
  qty            double precision NOT NULL DEFAULT 0,   -- filled size, always >= 0
  entry_price    double precision,
  stop_price     double precision,
  exit_due_at    timestamptz,
  status         text NOT NULL DEFAULT 'opening' CHECK (status IN ('opening', 'open', 'closing', 'closed', 'failed')),
  open_order_id  uuid REFERENCES orders(id),
  close_order_id uuid REFERENCES orders(id),
  close_reason   text,
  pnl            double precision,
  opened_at      timestamptz NOT NULL DEFAULT now(),
  closed_at      timestamptz
);
CREATE UNIQUE INDEX news_positions_one_per_symbol ON news_positions (book_id, symbol)
  WHERE status IN ('opening', 'open', 'closing');
CREATE INDEX news_positions_prediction_idx ON news_positions (prediction_id);
