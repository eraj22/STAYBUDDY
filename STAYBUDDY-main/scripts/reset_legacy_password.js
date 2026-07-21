const path = require("node:path");
const { Client } = require("pg");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const { hashPassword, isPassword } = require("../passwords");

const emailIndex = process.argv.indexOf("--email");
const email = emailIndex >= 0 ? process.argv[emailIndex + 1] : "";
const password = process.env.NEW_PASSWORD;

if (!email || !isPassword(password)) {
  console.error("Usage: set NEW_PASSWORD in your terminal, then run: node scripts/reset_legacy_password.js --email user@example.com");
  process.exitCode = 1;
  return;
}

const client = new Client({
  host: process.env.DB_HOST,
  port: Number(process.env.DB_PORT),
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
});

async function main() {
  await client.connect();
  try {
    const passwordHash = await hashPassword(password);
    const result = await client.query(
      "UPDATE users SET password_hash=$1 WHERE email=$2 RETURNING id, email, role",
      [passwordHash, email]
    );
    if (result.rowCount === 0) {
      throw new Error("No account found for that email");
    }
    console.log(`Password reset for ${result.rows[0].email}`);
  } finally {
    await client.end();
  }
}

main().catch((error) => {
  console.error(`Password reset failed: ${error.message}`);
  process.exitCode = 1;
});