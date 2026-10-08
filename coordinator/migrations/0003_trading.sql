-- Stage 4: paper trading through the coordinator.
-- Every model trades from its own book (its share of the one Alpaca account): the
-- coordinator alone places orders, sized against the book and the limits.
CREATE TABLE books (
  id               bigserial PRIMARY KEY,
  model_id         text NOT NULL REFERENCES models(id),
  mode             text NOT NULL CHECK (mode IN ('paper', 'live')),
  status           text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closing', 'closed')),
  starting_balance double precision NOT NULL,
  cash             double precision NOT NULL,
  job_id           uuid REFERENCES jobs(id),
  targets          jsonb,                 -- the model's latest wanted weights {symbol: weight}
  targets_bar_t    bigint,                -- epoch of the last bar the decision used
  targets_reason   text,
  targets_worker   text REFERENCES workers(id),
  targets_at       timestamptz,
  executed_bar_t   bigint,                -- the decision last turned into orders
  waiting          text,                  -- why the latest decision is not traded yet (shown on the dashboard)
  started_at       timestamptz NOT NULL DEFAULT now(),
  closed_at        timestamptz
);
CREATE UNIQUE INDEX books_one_open_per_model ON books (model_id) WHERE status IN ('active', 'closing');

CREATE TABLE positions (
  book_id   bigint NOT NULL REFERENCES books(id),
  symbol    text NOT NULL,
  qty       double precision NOT NULL,
  cost      double precision NOT NULL,     -- dollars paid for what is still held, costs included
  opened_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (book_id, symbol)
);

-- Every order, with the model, the worker whose signal caused it, the time and the reason.
CREATE TABLE orders (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),   -- also Alpaca's client_order_id
  book_id          bigint NOT NULL REFERENCES books(id),
  model_id         text NOT NULL REFERENCES models(id),
  worker_id        text REFERENCES workers(id),
  mode             text NOT NULL CHECK (mode IN ('paper', 'live')),
  symbol           text NOT NULL,
  side             text NOT NULL CHECK (side IN ('buy', 'sell')),
  notional         double precision,      -- dollars, for buys
  qty              double precision,      -- shares or coins, for sells
  reason           text NOT NULL,
  status           text NOT NULL CHECK (status IN ('submitting', 'submitted', 'partially_filled', 'filled',
                                                   'cancelled', 'rejected', 'blocked')),
  broker_order_id  text,
  filled_qty       double precision NOT NULL DEFAULT 0,
  filled_avg_price double precision,
  booked_qty       double precision NOT NULL DEFAULT 0,   -- how much of the fill is already in the book
  error            text,
  created_at       timestamptz NOT NULL DEFAULT now(),
  submitted_at     timestamptz,
  finished_at      timestamptz
);
CREATE INDEX orders_open_idx ON orders (created_at) WHERE status IN ('submitting', 'submitted', 'partially_filled');
CREATE INDEX orders_model_idx ON orders (model_id, created_at DESC);

-- Every decision a worker sent, kept or not.
CREATE TABLE signals (
  id         bigserial PRIMARY KEY,
  book_id    bigint NOT NULL REFERENCES books(id),
  model_id   text NOT NULL,
  worker_id  text,
  bar_t      bigint NOT NULL,
  targets    jsonb NOT NULL,
  reason     text,
  outcome    text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX signals_book_idx ON signals (book_id, id DESC);
