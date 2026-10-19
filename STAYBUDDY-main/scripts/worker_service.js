const http = require("node:http");
const { Pool } = require("pg");
const { runDueJobs, enqueueRecurringJobs } = require("../operational_job_worker");

const pool = new Pool({ host: process.env.DB_HOST, port: Number(process.env.DB_PORT), user: process.env.DB_USER, password: process.env.DB_PASSWORD, database: process.env.DB_NAME });
let state = { healthy: false, running: false, last_run_at: null, last_error: null, processed: 0 };

async function cycle() {
  if (state.running) return;
  state.running = true;
  try {
    await enqueueRecurringJobs(pool);
    const results = await runDueJobs(pool, { limit: Number(process.env.OPERATIONAL_JOB_LIMIT || 25) });
    state = { healthy: true, running: false, last_run_at: new Date().toISOString(), last_error: null, processed: results.length };
  } catch (error) {
    state = { ...state, healthy: false, running: false, last_run_at: new Date().toISOString(), last_error: error.message };
  }
}

http.createServer((_req,res)=>{res.writeHead(state.healthy?200:503,{"Content-Type":"application/json"});res.end(JSON.stringify(state));}).listen(Number(process.env.WORKER_HEALTH_PORT||5001));
cycle();
setInterval(cycle,Number(process.env.WORKER_INTERVAL_MS||15000));
for(const signal of ["SIGINT","SIGTERM"])process.on(signal,async()=>{await pool.end();process.exit(0);});
