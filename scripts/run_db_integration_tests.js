const path = require("node:path");
const { spawn } = require("node:child_process");
const { Client } = require("pg");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const sourceDatabase = process.env.DB_NAME;
const testDatabase = process.env.TEST_DB_NAME || "staybuddy_test";
const connection = {
  host: process.env.DB_HOST,
  port: Number(process.env.DB_PORT),
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
};

function quoteIdentifier(identifier) {
  return `"${identifier.replace(/"/g, '""')}"`;
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: "inherit", ...options });
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} exited with code ${code}`));
    });
  });
}

async function resetTestDatabase() {
  const maintenanceClient = new Client({ ...connection, database: "postgres" });
  await maintenanceClient.connect();
  try {
    await maintenanceClient.query(
      "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=$1 AND pid <> pg_backend_pid()",
      [testDatabase]
    );
    await maintenanceClient.query(`DROP DATABASE IF EXISTS ${quoteIdentifier(testDatabase)}`);
    await maintenanceClient.query(`CREATE DATABASE ${quoteIdentifier(testDatabase)}`);
  } finally {
    await maintenanceClient.end();
  }
}

async function restoreSourceDatabase() {
  const environment = { ...process.env, PGPASSWORD: process.env.DB_PASSWORD };
  const dumpArgs = [
    "--no-owner", "--no-privileges",
    "-h", connection.host,
    "-p", String(connection.port),
    "-U", connection.user,
    sourceDatabase,
  ];
  const restoreArgs = [
    "-v", "ON_ERROR_STOP=1",
    "-h", connection.host,
    "-p", String(connection.port),
    "-U", connection.user,
    "-d", testDatabase,
  ];

  await new Promise((resolve, reject) => {
    const dump = spawn("pg_dump", dumpArgs, { env: environment });
    const restore = spawn("psql", restoreArgs, { env: environment });
    let dumpError = "";
    let restoreError = "";

    dump.stderr.on("data", (chunk) => { dumpError += chunk; });
    restore.stderr.on("data", (chunk) => { restoreError += chunk; });
    dump.stdout.pipe(restore.stdin);
    dump.once("error", reject);
    restore.once("error", reject);
    restore.once("close", (restoreCode) => {
      if (restoreCode === 0) resolve();
      else reject(new Error(`Database restore failed: ${restoreError || dumpError}`));
    });
  });
}

async function dropTestDatabase() {
  const maintenanceClient = new Client({ ...connection, database: "postgres" });
  await maintenanceClient.connect();
  try {
    await maintenanceClient.query(
      "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=$1 AND pid <> pg_backend_pid()",
      [testDatabase]
    );
    await maintenanceClient.query(`DROP DATABASE IF EXISTS ${quoteIdentifier(testDatabase)}`);
  } finally {
    await maintenanceClient.end();
  }
}

async function main() {
  if (!sourceDatabase) throw new Error("DB_NAME must be configured");
  if (sourceDatabase === testDatabase || !testDatabase.startsWith("staybuddy_test")) {
    throw new Error("TEST_DB_NAME must start with staybuddy_test and differ from DB_NAME");
  }

  try {
    await resetTestDatabase();
    await restoreSourceDatabase();
    await run(process.execPath, ["--test", "--test-concurrency=1", "test/booking_capacity.integration.test.js", "test/owner_authorization.integration.test.js", "test/parent_authorization.integration.test.js", "test/attendance.integration.test.js", "test/emergency.integration.test.js", "test/verification.integration.test.js", "test/fees.integration.test.js", "test/messaging.integration.test.js", "test/warden_notifications.integration.test.js", "test/payments.integration.test.js"], {
      cwd: path.join(__dirname, ".."),
      env: {
        ...process.env,
        DB_NAME: testDatabase,
        RUN_DB_TESTS: "1",
        TEST_PASSWORD_RESET_TOKENS: "1",
        TEST_PARENT_INVITATION_TOKENS: "1",
        TEST_API_PORT: process.env.TEST_API_PORT || "5002",
      },
    });
  } finally {
    await dropTestDatabase();
  }
}

main().catch((error) => {
  console.error(`Database integration tests failed: ${error.message}`);
  process.exitCode = 1;
});
