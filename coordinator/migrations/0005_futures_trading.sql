-- Stage 6: futures models trading. First on Alpaca paper (SPY and QQQ shares standing in
-- for MES and MNQ contracts), then on Topstep (real contracts through TopstepX) once a
-- model is ready for a Combine. Separate from the stock and crypto books, which keep
-- their own tables.

-- The latest 1-minute prices for live decisions, per source: "alpaca" (SPY/QQQ scaled to
-- index points, for Alpaca paper), "topstepx" (the real contract, for Topstep) or
-- "synthetic" (demo). Only the last couple of months are kept.
CREATE TABLE live_bars (
  source        text NOT NULL,
  symbol        text NOT NULL,             -- MES or MNQ
  ts            timestamptz NOT NULL,      -- bar start (UTC)
  open          double precision NOT NULL,
  high          double precision NOT NULL,
  low           double precision NOT NULL,
  close         double precision NOT NULL,
  volume        double precision NOT NULL,
  instrument_id bigint NOT NULL DEFAULT 0,
  PRIMARY KEY (source, symbol, ts)
);

-- One book per model and place it trades. contracts: the model's full size in micros.
CREATE TABLE futures_books (
  id             bigserial PRIMARY KEY,
  model_id       text NOT NULL REFERENCES models(id),
  venue          text NOT NULL CHECK (venue IN ('alpaca_paper', 'topstep')),
  status         text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closing', 'closed')),
  contracts      integer NOT NULL CHECK (contracts >= 1),
  account        text,                     -- the TopstepX account id (Topstep only)
  job_id         uuid REFERENCES jobs(id),
  target         jsonb,                    -- {symbol: contracts} the model wants now (+ long, - short)
  target_t       bigint,                   -- epoch of the last minute the decision used
  target_reason  text,
  target_worker  text REFERENCES workers(id),
  target_at      timestamptz,
  waiting        text,                     -- why the book is not where the model wants it
  started_at     timestamptz NOT NULL DEFAULT now(),
  closed_at      timestamptz
);
CREATE UNIQUE INDEX futures_books_one_open ON futures_books (model_id, venue) WHERE status IN ('active', 'closing');

-- What each book holds, in the traded instrument's units (shares of SPY/QQQ, or contracts),
-- + long, - short, at an average price in that instrument's own prices.
CREATE TABLE futures_positions (
  book_id    bigint NOT NULL REFERENCES futures_books(id),
  symbol     text NOT NULL,                -- MES or MNQ (what the model trades)
  instrument text NOT NULL,                -- SPY, QQQ, or the TopstepX contract id
  qty        double precision NOT NULL,
  avg_price  double precision NOT NULL,
  PRIMARY KEY (book_id, symbol)
);

-- Every futures order, with the model, the worker whose decision caused it, the time and
-- the reason; blocked ones too.
CREATE TABLE futures_orders (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),   -- also the broker's client order id / tag
  book_id          bigint NOT NULL REFERENCES futures_books(id),
  model_id         text NOT NULL REFERENCES models(id),
  worker_id        text REFERENCES workers(id),
  venue            text NOT NULL,
  symbol           text NOT NULL,
  instrument       text NOT NULL,
  side             text NOT NULL CHECK (side IN ('buy', 'sell')),
  qty              double precision NOT NULL,                    -- instrument units
  contracts        double precision NOT NULL,                    -- the same, in micro contracts
  reason           text NOT NULL,
  status           text NOT NULL CHECK (status IN ('submitting', 'submitted', 'partially_filled', 'filled',
                                                   'cancelled', 'rejected', 'blocked')),
  broker_order_id  text,
  filled_qty       double precision NOT NULL DEFAULT 0,
  filled_avg_price double precision,
  booked_qty       double precision NOT NULL DEFAULT 0,
  error            text,
  created_at       timestamptz NOT NULL DEFAULT now(),
  submitted_at     timestamptz,
  finished_at      timestamptz
);
CREATE INDEX futures_orders_open_idx ON futures_orders (created_at)
  WHERE status IN ('submitting', 'submitted', 'partially_filled');
CREATE INDEX futures_orders_book_idx ON futures_orders (book_id, created_at DESC);

-- Each book's results per trading day in futures dollars (commission included, also on
-- Alpaca paper, where it is not charged, so the numbers compare with the backtest).
CREATE TABLE futures_days (
  book_id  bigint NOT NULL REFERENCES futures_books(id),
  day      date NOT NULL,
  pnl      double precision NOT NULL DEFAULT 0,
  fees     double precision NOT NULL DEFAULT 0,
  dip      double precision NOT NULL DEFAULT 0,
  trades   integer NOT NULL DEFAULT 0,
  PRIMARY KEY (book_id, day)
);

-- The Topstep account's balance at the end of each trading day, for the loss floor.
CREATE TABLE topstep_days (
  account  text NOT NULL,
  day      date NOT NULL,
  balance  double precision NOT NULL,
  PRIMARY KEY (account, day)
);

INSERT INTO settings (key, value) VALUES ('topstep_paused', 'false'), ('topstep_paused_reason', 'null'),
  ('topstep_confirmed', 'false');
