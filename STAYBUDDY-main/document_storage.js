const crypto = require("node:crypto");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");

function createLocalDocumentStorage({ directory = process.env.PRIVATE_DOCUMENTS_DIR || path.join(process.cwd(), ".private-documents") } = {}) {
  const root = path.resolve(directory);

  function objectPath(key) {
    if (typeof key !== "string" || !/^(verification|complaint|fee_receipt|student_payment_proof|message_attachment|booking_document)\/[a-zA-Z0-9_-]+\/[a-f0-9-]{36}$/.test(key)) throw new Error("Invalid private document key");
    const resolved = path.resolve(root, ...key.split("/"));
    if (!resolved.startsWith(`${root}${path.sep}`)) throw new Error("Invalid private document key");
    return resolved;
  }

  return {
    provider: "local",
    createKey(scopeOrId, maybeId) {
      const scope = maybeId == null ? "verification" : String(scopeOrId);
      const ownerId = maybeId == null ? scopeOrId : maybeId;
      if (!/^(verification|complaint|fee_receipt|student_payment_proof|message_attachment|booking_document)$/.test(scope) || !Number.isInteger(Number(ownerId)) || Number(ownerId) <= 0) throw new Error("Invalid private document owner");
      return `${scope}/${ownerId}/${crypto.randomUUID()}`;
    },
    async save(key, bytes) {
      const target = objectPath(key);
      await fsp.mkdir(path.dirname(target), { recursive: true });
      const handle = await fsp.open(target, "wx", 0o600);
      try { await handle.writeFile(bytes); } finally { await handle.close(); }
    },
    async remove(key) {
      try { await fsp.unlink(objectPath(key)); } catch (error) { if (error.code !== "ENOENT") throw error; }
    },
    async exists(key) {
      try { await fsp.access(objectPath(key)); return true; } catch (_) { return false; }
    },
    createReadStream(key) { return fs.createReadStream(objectPath(key)); },
  };
}

module.exports = { createLocalDocumentStorage };
