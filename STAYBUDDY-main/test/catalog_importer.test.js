const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const {
  calculateAvailableCapacity,
  normalizeRecord,
  parseCsv,
} = require("../scripts/import_hostel_catalog");

const sourcePath = path.join(__dirname, "..", "ml-notebooks", "data", "hostels.csv");

test("the catalog CSV parses into 75 valid unique hostel records", () => {
  const rows = parseCsv(fs.readFileSync(sourcePath, "utf8"));
  const records = rows.map(normalizeRecord);
  const externalIds = new Set(records.map((record) => record.externalId));

  assert.equal(records.length, 75);
  assert.equal(externalIds.size, records.length);
  assert.ok(records.every((record) => record.availableCapacity <= record.totalCapacity));
  assert.ok(records.every((record) => Array.isArray(record.amenities)));
});

test("quoted CSV fields preserve commas and JSON arrays", () => {
  const rows = parseCsv(fs.readFileSync(sourcePath, "utf8"));
  const firstRecord = normalizeRecord(rows[0]);

  assert.match(firstRecord.address, /Street 15/);
  assert.deepEqual(firstRecord.roomTypes, ["Single", "Double"]);
});

test("live availability subtracts active bookings and rejects overbooking", () => {
  assert.equal(calculateAvailableCapacity(13, 0, "HST-001"), 13);
  assert.equal(calculateAvailableCapacity(13, 1, "HST-001"), 12);
  assert.throws(
    () => calculateAvailableCapacity(2, 3, "HST-002"),
    /HST-002: active bookings exceed imported available capacity/
  );
});