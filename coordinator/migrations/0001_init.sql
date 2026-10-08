-- fleet-v2 schema, stage 1: fleet core (adapted from polymarket-fleet 0001_init.sql).
-- Changes from v1: no worker roles or epochs; jobs carry a one-line detail, a nullable
-- progress (null = "Live", paper trading) and the worker that last ran them; a job on a
-- worker that goes offline is failed (never silently requeued) and can be run again.
CREATE TABLE IF NOT EXISTS schema_migrations (
  version     text PRIMARY KEY,
  applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE settings (
  key         text PRIMARY KEY,
  value       jsonb NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now()
);
INSERT INTO settings (key, value) VALUES
  ('lease_seconds',         '30'),
  ('heartbeat_seconds',     '5'),
  ('online_after_seconds',  '20'),
  ('trading_paused',        'false'),
  ('paused_reason',         'null'),
  ('live_confirmed',        'false');

CREATE TABLE workers (
  id                text PRIMARY KEY,
  name              text NOT NULL,
  token_hash        text NOT NULL,
  prev_token_hash   text,
  enabled           boolean NOT NULL DEFAULT true,
  cpu_pct           real,
  ram_pct           real,
  ram_used_mb       integer,
  ram_total_mb      integer,
  temp_c            real,
  skew_ms           integer,
  python_version    text,
  code_version      text,
  hostname          text,
  boot_id           text,
  remote_ip         text,
  registered_at     timestamptz NOT NULL DEFAULT now(),
  last_heartbeat_at timestamptz
);
CREATE UNIQUE INDEX workers_name_idx ON workers (name);

CREATE TABLE enroll_tokens (
  token_hash        text PRIMARY KEY,
  created_at        timestamptz NOT NULL DEFAULT now(),
  expires_at        timestamptz NOT NULL,
  used_by_worker_id text REFERENCES workers(id),
  used_at           timestamptz
);

CREATE TABLE jobs (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kind              text NOT NULL CHECK (kind IN ('sleep','data_refresh','backtest','paper_trade','model_search')),
  status            text NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued','leased','cancel_requested','succeeded','failed','cancelled')),
  params            jsonb NOT NULL DEFAULT '{}'::jsonb,
  model_id          text,
  checkpoint        jsonb,
  progress          real DEFAULT 0,
  detail            text,
  target_worker_id  text REFERENCES workers(id),
  target_auto       boolean NOT NULL DEFAULT false,
  lease_worker_id   text REFERENCES workers(id),
  last_worker_id    text REFERENCES workers(id),
  lease_token       uuid,
  lease_expires_at  timestamptz,
  preempt_requested boolean NOT NULL DEFAULT false,
  run_after         timestamptz NOT NULL DEFAULT now(),
  idempotency_key   text UNIQUE,
  retry_of          uuid REFERENCES jobs(id),
  result            jsonb,
  error             text,
  created_at        timestamptz NOT NULL DEFAULT now(),
  started_at        timestamptz,
  finished_at       timestamptz,
  updated_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX jobs_queued_idx ON jobs (created_at) WHERE status = 'queued';
CREATE INDEX jobs_leased_idx ON jobs (lease_expires_at) WHERE status IN ('leased','cancel_requested');
CREATE INDEX jobs_lease_worker_idx ON jobs (lease_worker_id) WHERE status IN ('leased','cancel_requested');
CREATE INDEX jobs_target_idx ON jobs (target_worker_id) WHERE status IN ('queued','leased','cancel_requested');
CREATE INDEX jobs_finished_idx ON jobs (finished_at DESC) WHERE finished_at IS NOT NULL;

CREATE TABLE job_events (
  id        bigserial PRIMARY KEY,
  job_id    uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  ts        timestamptz NOT NULL DEFAULT now(),
  worker_id text,
  event     text NOT NULL,
  detail    jsonb
);
CREATE INDEX job_events_job_idx ON job_events (job_id, id);

CREATE TABLE audit_log (
  id                bigserial PRIMARY KEY,
  ts                timestamptz NOT NULL DEFAULT now(),
  actor             text,
  ip                text,
  action            text NOT NULL,
  entity            text,
  before            jsonb,
  after             jsonb,
  confirmation_text text
);
