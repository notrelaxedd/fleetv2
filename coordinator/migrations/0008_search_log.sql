-- Search log: every setting a model search tried, kept or not, so the searches can learn
-- from their mistakes (what was tried, how it scored, why it was rejected) and give up on
-- model files and recipes that keep failing (coordinator/search.py).
--
-- family: the model file's name (stocks, crypto, futures) or a recipe's name (recipe_...).
-- score: the TRAINING score (null when it never scored, e.g. too few trades). rejected:
-- the plain reason a setting was thrown away, if it was. metrics: compact numbers only
-- (never curves or daily arrays). Nothing from the held-out period or the lockbox is ever
-- written here. job_id is not a foreign key: the log outlives the jobs.
CREATE TABLE search_log (
  id          bigserial PRIMARY KEY,
  market      text NOT NULL CHECK (market IN ('stocks', 'crypto', 'futures')),
  family      text NOT NULL,
  job_id      uuid,
  worker_id   text,
  params      jsonb NOT NULL,
  score       double precision,
  rejected    text,
  metrics     jsonb,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX search_log_best ON search_log (market, family, score DESC NULLS LAST);
CREATE INDEX search_log_recent ON search_log (market, family, created_at DESC, id DESC);
