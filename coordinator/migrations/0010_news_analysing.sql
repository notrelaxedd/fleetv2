-- News engine: an article is "analysing" while Claude reads it (claimed before the call,
-- so it can never go back to "pending" and be paid for twice). One left "analysing" by a
-- crash is turned into "failed" by the next analyst pass after STALE_CLAIM.
ALTER TABLE news DROP CONSTRAINT news_analysis_status_check;
ALTER TABLE news ADD CONSTRAINT news_analysis_status_check CHECK (analysis_status IN
  ('pending', 'backfill', 'analysing', 'analysed', 'skipped', 'failed', 'over_budget'));
ALTER TABLE news ADD COLUMN claimed_at timestamptz;   -- when the analyst claimed it
