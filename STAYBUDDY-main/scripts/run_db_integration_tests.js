const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { Client } = require("pg");

const projectRoot = path.join(__dirname, "..");
require("dotenv").config({ path: path.join(projectRoot, ".env") });

// SQL sources are UTF-8 even when Windows psql defaults to WIN1252.
process.env.PGCLIENTENCODING = "UTF8";

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

function selectIntegrationTests() {
  const allTests = fs.readdirSync(path.join(projectRoot, "test"))
    .filter((filename) => filename.endsWith(".integration.test.js"))
    .sort((left, right) => left.localeCompare(right));
  const requested = String(process.env.DB_TEST_FILES || "")
    .split(",")
    .map((filename) => path.basename(filename.trim()))
    .filter(Boolean);
  if (!requested.length) return allTests.map((filename) => `test/${filename}`);

  const unknown = requested.filter((filename) => !allTests.includes(filename));
  if (unknown.length) throw new Error(`Unknown integration test file(s): ${unknown.join(", ")}`);
  return requested.map((filename) => `test/${filename}`);
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

async function applyMigration(filename) {
  await run(
    "psql",
    ["-v", "ON_ERROR_STOP=1", "-h", connection.host, "-p", String(connection.port), "-U", connection.user, "-d", testDatabase, "-f", path.join("migrations", filename)],
    {
      cwd: projectRoot,
      env: { ...process.env, PGPASSWORD: process.env.DB_PASSWORD },
      stdio: ["ignore", "ignore", "inherit"],
    }
  );
}

async function installFreshSchema() {
  await run(
    "psql",
    ["-v", "ON_ERROR_STOP=1", "-h", connection.host, "-p", String(connection.port), "-U", connection.user, "-d", testDatabase, "-f", "staybuddy_schema.sql"],
    {
      cwd: projectRoot,
      env: { ...process.env, PGPASSWORD: process.env.DB_PASSWORD },
      stdio: ["ignore", "ignore", "inherit"],
    }
  );

  const migrations = fs.readdirSync(path.join(projectRoot, "migrations"))
    .filter((filename) => /^\d{3}_.+\.sql$/.test(filename))
    .sort((left, right) => left.localeCompare(right));
  for (const filename of migrations) await applyMigration(filename);

  const testEnvironment = { ...process.env, DB_NAME: testDatabase, RUN_DB_TESTS: "1" };
  await run(process.execPath, ["scripts/import_hostel_catalog.js"], { cwd: projectRoot, env: testEnvironment });
  await run(
    "psql",
    [
      "-v", "ON_ERROR_STOP=1", "-h", connection.host, "-p", String(connection.port), "-U", connection.user, "-d", testDatabase, "-c",
      "UPDATE hostels SET publication_status='published' WHERE verified=true; INSERT INTO rooms(hostel_id,room_number,capacity,available_capacity,room_type,monthly_rent) SELECT h.id,'CATALOG-1',h.available_capacity,h.available_capacity,'shared',COALESCE(h.single_room_price,h.double_room_price,h.dorm_room_price,0) FROM hostels h WHERE h.source='ml_catalog' AND h.available_capacity>0 AND NOT EXISTS(SELECT 1 FROM rooms r WHERE r.hostel_id=h.id);",
    ],
    {
      cwd: projectRoot,
      env: { ...process.env, PGPASSWORD: process.env.DB_PASSWORD },
      stdio: ["ignore", "ignore", "inherit"],
    }
  );
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

  const integrationTests = selectIntegrationTests();
  console.log(`Running ${integrationTests.length} integration test file(s) on disposable database ${testDatabase}`);
  try {
    await resetTestDatabase();
    await installFreshSchema();
    await run(process.execPath, ["--test", "--test-concurrency=1", ...integrationTests], {
      cwd: projectRoot,
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
