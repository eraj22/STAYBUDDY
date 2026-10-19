const crypto = require("crypto");

async function enqueueOperationalJob(client, jobType, dedupeKey, payload = {}, options = {}) {
  if (!jobType || !dedupeKey) throw new Error("jobType and dedupeKey are required");
  const result = await client.query(
    `INSERT INTO operational_jobs (job_type,dedupe_key,payload,run_at,max_attempts)
     VALUES ($1,$2,$3::jsonb,COALESCE($4::timestamptz,NOW()),COALESCE($5,5))
     ON CONFLICT (dedupe_key) DO NOTHING
     RETURNING *`,
    [jobType, dedupeKey, JSON.stringify(payload), options.runAt || null, options.maxAttempts || null]
  );
  return { job: result.rows[0] || null, duplicate: result.rowCount === 0 };
}

function deliveryDedupeKey(kind, deliveryId) {
  return `${kind}:${deliveryId}:${crypto.createHash("sha256").update(String(deliveryId)).digest("hex").slice(0, 12)}`;
}

module.exports = { enqueueOperationalJob, deliveryDedupeKey };
