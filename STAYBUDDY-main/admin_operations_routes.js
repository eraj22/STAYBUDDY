function boundedLimit(value, fallback = 50) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? Math.min(parsed, 100) : fallback;
}

function validJobId(value) {
  const id = Number(value);
  return Number.isSafeInteger(id) && id > 0 ? id : null;
}

function registerAdminOperationsRoutes({ app, pool, requireAuth, requireRole }) {
  app.get("/api/admin/platform/health", requireAuth, requireRole("admin"), async (_req, res) => {
    try {
      const [users, hostels, bookings, complaints, emergencies, applications, jobs] = await Promise.all([
        pool.query("SELECT count(*)::int AS total, COALESCE(jsonb_object_agg(role,count), '{}'::jsonb) AS by_role FROM (SELECT role,count(*)::int AS count FROM users GROUP BY role) grouped"),
        pool.query("SELECT count(*)::int AS total, count(*) FILTER (WHERE verified)::int AS verified, count(*) FILTER (WHERE NOT verified)::int AS unverified FROM hostels"),
        pool.query("SELECT count(*) FILTER (WHERE status IN ('pending','confirmed'))::int AS active, count(*)::int AS total FROM bookings"),
        pool.query("SELECT count(*) FILTER (WHERE status NOT IN ('resolved','closed'))::int AS open, count(*) FILTER (WHERE status NOT IN ('resolved','closed') AND sla_due_at IS NOT NULL AND sla_due_at<NOW())::int AS overdue_sla, count(*)::int AS total FROM complaints"),
        pool.query("SELECT count(*) FILTER (WHERE status IN ('active','acknowledged','escalated'))::int AS active, count(*)::int AS total FROM emergency_incidents"),
        pool.query("SELECT COALESCE(jsonb_object_agg(status,count), '{}'::jsonb) AS by_status, count(*)::int AS total FROM (SELECT status,count(*)::int AS count FROM hostel_registration_applications GROUP BY status) grouped"),
        pool.query(`SELECT COALESCE((SELECT jsonb_object_agg(status,count) FROM (SELECT status,count(*)::int AS count FROM operational_jobs GROUP BY status) grouped), '{}'::jsonb) AS by_status,
          count(*) FILTER (WHERE status='queued' AND run_at<=NOW())::int AS due, count(*)::int AS total FROM operational_jobs`)
      ]);
      res.json({ generated_at: new Date().toISOString(), users: users.rows[0], hostels: hostels.rows[0], bookings: bookings.rows[0], complaints: complaints.rows[0], emergencies: emergencies.rows[0], verification_applications: applications.rows[0], operational_jobs: jobs.rows[0] });
    } catch (error) { res.status(500).json({ error: "Failed to load platform health" }); }
  });

  app.get("/api/admin/platform/jobs", requireAuth, requireRole("admin"), async (req, res) => {
    const allowed = new Set(["queued", "processing", "succeeded", "failed", "dead"]); const status = req.query.status ? String(req.query.status) : null;
    if (status && !allowed.has(status)) return res.status(400).json({ error: "Invalid job status" });
    try {
      const result = await pool.query(`SELECT id,job_type,status,run_at,attempt_count,max_attempts,locked_at,locked_by,last_error,completed_at,created_at,updated_at
        FROM operational_jobs WHERE ($1::varchar IS NULL OR status=$1) ORDER BY created_at DESC LIMIT $2`, [status, boundedLimit(req.query.limit)]);
      res.json({ jobs: result.rows });
    } catch (error) { res.status(500).json({ error: "Failed to load operational jobs" }); }
  });

  app.get("/api/admin/platform/jobs/:id/attempts", requireAuth, requireRole("admin"), async (req, res) => {
    const jobId = validJobId(req.params.id);
    if (!jobId) return res.status(400).json({ error: "Invalid job id" });
    try {
      const [job, attempts, retries] = await Promise.all([
        pool.query(`SELECT id,job_type,status,run_at,attempt_count,max_attempts,locked_at,locked_by,last_error,completed_at,created_at,updated_at
          FROM operational_jobs WHERE id=$1`, [jobId]),
        pool.query("SELECT attempt_number,outcome,details,created_at FROM operational_job_attempts WHERE job_id=$1 ORDER BY attempt_number DESC", [jobId]),
        pool.query(`SELECT id,retried_by_user_id,reason,previous_attempt_count,previous_max_attempts,retry_max_attempts,retried_at
          FROM operational_job_manual_retries WHERE job_id=$1 ORDER BY retried_at DESC`, [jobId]),
      ]);
      if (!job.rowCount) return res.status(404).json({ error: "Operational job not found" });
      res.json({ job: job.rows[0], attempts: attempts.rows, manual_retries: retries.rows });
    } catch (error) { res.status(500).json({ error: "Failed to load operational job history" }); }
  });

  app.post("/api/admin/platform/jobs/:id/retry", requireAuth, requireRole("admin"), async (req, res) => {
    const jobId = validJobId(req.params.id);
    const reason = String(req.body?.reason || "").trim();
    if (!jobId) return res.status(400).json({ error: "Invalid job id" });
    if (!reason || reason.length > 2000) return res.status(400).json({ error: "A retry reason of up to 2000 characters is required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const job = await client.query("SELECT * FROM operational_jobs WHERE id=$1 FOR UPDATE", [jobId]);
      if (!job.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Operational job not found" }); }
      const current = job.rows[0];
      if (current.status !== "dead") { await client.query("ROLLBACK"); return res.status(409).json({ error: "Only dead operational jobs can be retried manually" }); }
      if (Number(current.max_attempts) >= 20) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Operational job has reached the manual retry limit" }); }
      const nextMaxAttempts = Number(current.max_attempts) + 1;
      const retried = await client.query(`UPDATE operational_jobs
        SET status='queued', run_at=NOW(), max_attempts=$1, locked_at=NULL, locked_by=NULL, last_error=NULL, completed_at=NULL, updated_at=NOW()
        WHERE id=$2 RETURNING id,job_type,status,run_at,attempt_count,max_attempts,locked_at,locked_by,last_error,completed_at,created_at,updated_at`, [nextMaxAttempts, jobId]);
      const audit = await client.query(`INSERT INTO operational_job_manual_retries
        (job_id,retried_by_user_id,reason,previous_attempt_count,previous_max_attempts,retry_max_attempts)
        VALUES ($1,$2,$3,$4,$5,$6) RETURNING *`, [jobId, req.user.id, reason, current.attempt_count, current.max_attempts, nextMaxAttempts]);
      await client.query("COMMIT");
      res.json({ job: retried.rows[0], manual_retry: audit.rows[0] });
    } catch (error) {
      await client.query("ROLLBACK");
      res.status(500).json({ error: "Failed to retry operational job" });
    } finally { client.release(); }
  });
}

module.exports = { registerAdminOperationsRoutes };
