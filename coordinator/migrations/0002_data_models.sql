-- Stage 2: cached price bars (coordinator only) and models with their backtest results.
CREATE TABLE bars (
  symbol     text NOT NULL,
  timeframe  text NOT NULL CHECK (timeframe IN ('1Day', '1Hour')),
  ts         timestamptz NOT NULL,          -- bar start time (UTC)
  open       double precision NOT NULL,
  high       double precision NOT NULL,
  low        double precision NOT NULL,
  close      double precision NOT NULL,
  volume     double precision NOT NULL,
  feed       text NOT NULL,                 -- iex, sip (stocks) or crypto
  PRIMARY KEY (symbol, timeframe, ts)
);

-- One row per symbol: how far the cache reaches and when it was last refreshed.
CREATE TABLE bar_status (
  symbol       text NOT NULL,
  timeframe    text NOT NULL,
  first_ts     timestamptz,
  last_ts      timestamptz,
  bars         integer NOT NULL DEFAULT 0,
  feed         text,
  refreshed_at timestamptz,
  error        text,
  PRIMARY KEY (symbol, timeframe)
);

CREATE TABLE models (
  id            text PRIMARY KEY,           -- slug, e.g. momentum, momentum-s1-07
  name          text NOT NULL,
  module        text NOT NULL,              -- model file under fleet2/models (the code)
  market        text NOT NULL CHECK (market IN ('stocks', 'crypto')),
  description   text NOT NULL,              -- one sentence
  how_it_works  text NOT NULL,              -- three sentences, no jargon
  params        jsonb NOT NULL DEFAULT '{}'::jsonb,
  status        text CHECK (status IN ('backtested', 'paper_trading', 'retired')),
  origin        text NOT NULL DEFAULT 'starter' CHECK (origin IN ('starter', 'search')),
  metrics       jsonb,                      -- the last backtest (training and held-out)
  backtested_at timestamptz,
  created_at    timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE jobs ADD CONSTRAINT jobs_model_fk FOREIGN KEY (model_id) REFERENCES models(id);
