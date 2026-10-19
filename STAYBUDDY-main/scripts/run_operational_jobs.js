const path = require("node:path");
const { Pool } = require("pg");
const { runDueJobs, enqueueRecurringJobs } = require("../operational_job_worker");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

async function main() {
  const pool = new Pool({ host: process.env.DB_HOST, port: Number(process.env.DB_PORT), user: process.env.DB_USER, password: process.env.DB_PASSWORD, database: process.env.DB_NAME });
  try {
    await enqueueRecurringJobs(pool);
    const results = await runDueJobs(pool, { limit: Number(process.env.OPERATIONAL_JOB_LIMIT || 25) });
    console.log(`Processed ${results.length} operational job(s).`);
    for (const result of results) console.log(`${result.id}: ${result.type} - ${result.delivered ? "succeeded" : "retry queued"}`);
  } finally { await pool.end(); }
}

main().catch((error) => { console.error(`Operational jobs failed: ${error.message}`); process.exitCode = 1; });
