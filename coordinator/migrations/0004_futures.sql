-- Futures (Topstep day trading research): 1-minute bars of MES and MNQ, futures models,
-- the once-only Final check, and the count of settings model search has tried.
-- Stocks and crypto keep their tables and values; this only widens what is allowed.

ALTER TABLE bars DROP CONSTRAINT bars_timeframe_check;
ALTER TABLE bars ADD CONSTRAINT bars_timeframe_check CHECK (timeframe IN ('1Day', '1Hour', '1Min'));
-- The futures contract each bar came from (Databento's instrument id; 0 for stand-in
-- prices). Models are flat every night, so a day only ever uses one contract, and a
-- feature that compares two days skips a day whose contract changed (a roll).
ALTER TABLE bars ADD COLUMN instrument_id bigint;

ALTER TABLE models DROP CONSTRAINT models_market_check;
ALTER TABLE models ADD CONSTRAINT models_market_check CHECK (market IN ('stocks', 'crypto', 'futures'));

ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check
  CHECK (kind IN ('sleep', 'data_refresh', 'backtest', 'paper_trade', 'model_search', 'final_check'));

-- The lockbox: one Final check per model, kept forever. Rows can be added, never changed
-- or deleted (the trigger refuses), so a disappointing result cannot be re-run away.
CREATE TABLE final_checks (
  model_id    text PRIMARY KEY REFERENCES models(id),
  job_id      uuid REFERENCES jobs(id),
  worker_id   text REFERENCES workers(id),
  feed        text NOT NULL,
  result      jsonb NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE FUNCTION final_checks_are_forever() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'a Final check is stored forever and cannot be changed or deleted';
END $$;
CREATE TRIGGER final_checks_forever BEFORE UPDATE OR DELETE ON final_checks
  FOR EACH ROW EXECUTE FUNCTION final_checks_are_forever();

-- How many settings model search has tried per futures model file, and the sum and sum
-- of squares of their training Sharpe ratios: the "chance this is luck" figure needs both.
CREATE TABLE search_tries (
  module      text PRIMARY KEY,
  tries       bigint NOT NULL DEFAULT 0,
  sharpe_sum  double precision NOT NULL DEFAULT 0,
  sharpe_sq   double precision NOT NULL DEFAULT 0,
  updated_at  timestamptz NOT NULL DEFAULT now()
);
