const crypto = require("crypto");
const { promisify } = require("util");

const scrypt = promisify(crypto.scrypt);
const SCRYPT_COST = 16384;
const SCRYPT_BLOCK_SIZE = 8;
const SCRYPT_PARALLELIZATION = 1;
const KEY_LENGTH = 64;

function isPassword(value) {
  return typeof value === "string" && value.length >= 8;
}

async function hashPassword(password) {
  if (!isPassword(password)) {
    throw new Error("Password must be at least 8 characters long");
  }

  const salt = crypto.randomBytes(16).toString("base64url");
  const derivedKey = await scrypt(password, salt, KEY_LENGTH, {
    N: SCRYPT_COST,
    r: SCRYPT_BLOCK_SIZE,
    p: SCRYPT_PARALLELIZATION,
  });
  return [
    "scrypt",
    SCRYPT_COST,
    SCRYPT_BLOCK_SIZE,
    SCRYPT_PARALLELIZATION,
    salt,
    derivedKey.toString("base64url"),
  ].join("$");
}

async function verifyPassword(password, storedHash) {
  if (!isPassword(password) || typeof storedHash !== "string") return false;
  const parts = storedHash.split("$");
  if (parts.length !== 6 || parts[0] !== "scrypt") return false;

  const [algorithm, cost, blockSize, parallelization, salt, encodedKey] = parts;
  if (algorithm !== "scrypt") return false;
  const expectedKey = Buffer.from(encodedKey, "base64url");
  if (expectedKey.length === 0) return false;

  try {
    const derivedKey = await scrypt(password, salt, expectedKey.length, {
      N: Number(cost),
      r: Number(blockSize),
      p: Number(parallelization),
    });
    return derivedKey.length === expectedKey.length && crypto.timingSafeEqual(derivedKey, expectedKey);
  } catch {
    return false;
  }
}

module.exports = { hashPassword, isPassword, verifyPassword };