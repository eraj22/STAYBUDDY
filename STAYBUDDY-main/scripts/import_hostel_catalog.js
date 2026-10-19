const fs = require("fs");
const path = require("path");
const { Client } = require("pg");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const sourcePath = path.join(__dirname, "..", "ml-notebooks", "data", "hostels.csv");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];

    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        value += character;
      }
      continue;
    }

    if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(value);
      value = "";
    } else if (character === "\n") {
      row.push(value.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      value = "";
    } else {
      value += character;
    }
  }

  if (quoted) throw new Error("CSV contains an unclosed quoted value");
  if (value.length > 0 || row.length > 0) {
    row.push(value.replace(/\r$/, ""));
    rows.push(row);
  }

  const [headers, ...dataRows] = rows;
  return dataRows
    .filter((dataRow) => dataRow.some((cell) => cell.trim() !== ""))
    .map((dataRow, rowIndex) => {
      if (dataRow.length !== headers.length) {
        throw new Error(`CSV row ${rowIndex + 2} has ${dataRow.length} values; expected ${headers.length}`);
      }
      return Object.fromEntries(headers.map((header, columnIndex) => [header, dataRow[columnIndex]]));
    });
}

function required(value, field, externalId) {
  if (value == null || String(value).trim() === "") {
    throw new Error(`${externalId || "row"}: ${field} is required`);
  }
  return String(value).trim();
}

function integer(value, field, externalId, allowBlank = false) {
  if (allowBlank && (value == null || String(value).trim() === "")) return null;
  const parsed = Number.parseInt(required(value, field, externalId), 10);
  if (!Number.isInteger(parsed)) throw new Error(`${externalId}: ${field} must be an integer`);
  return parsed;
}

function decimal(value, field, externalId, allowBlank = false) {
  if (allowBlank && (value == null || String(value).trim() === "")) return null;
  const parsed = Number.parseFloat(required(value, field, externalId));
  if (!Number.isFinite(parsed)) throw new Error(`${externalId}: ${field} must be numeric`);
  return parsed;
}

function boolean(value, field, externalId) {
  const parsed = integer(value, field, externalId);
  if (parsed !== 0 && parsed !== 1) throw new Error(`${externalId}: ${field} must be 0 or 1`);
  return parsed === 1;
}

function jsonArray(value, field, externalId) {
  try {
    const parsed = JSON.parse(required(value, field, externalId));
    if (!Array.isArray(parsed)) throw new Error("not an array");
    return parsed;
  } catch {
    throw new Error(`${externalId}: ${field} must be a JSON array`);
  }
}

function normalizeRecord(row) {
  const externalId = required(row.hostel_id, "hostel_id");
  const totalCapacity = integer(row.total_capacity, "total_capacity", externalId);
  const availableCapacity = integer(row.available_rooms, "available_rooms", externalId);

  if (totalCapacity <= 0 || availableCapacity < 0 || availableCapacity > totalCapacity) {
    throw new Error(`${externalId}: capacity values are invalid`);
  }

  return {
    externalId,
    name: required(row.hostel_name, "hostel_name", externalId),
    hostelType: required(row.hostel_type, "hostel_type", externalId),
    verified: boolean(row.verified, "verified", externalId),
    yearEstablished: integer(row.year_established, "year_established", externalId),
    area: required(row.area, "area", externalId),
    city: required(row.city, "city", externalId),
    address: required(row.hostel_address, "hostel_address", externalId),
    latitude: decimal(row.latitude, "latitude", externalId),
    longitude: decimal(row.longitude, "longitude", externalId),
    distanceFromFastKm: decimal(row.distance_from_fast_km, "distance_from_fast_km", externalId),
    priceTier: required(row.price_tier, "price_tier", externalId),
    singleRoomPrice: integer(row.single_room_price, "single_room_price", externalId),
    doubleRoomPrice: integer(row.double_room_price, "double_room_price", externalId),
    dormRoomPrice: integer(row.dorm_room_price, "dorm_room_price", externalId),
    electricityIncluded: boolean(row.electricity_included, "electricity_included", externalId),
    electricityBillEstimate: integer(row.electricity_bill_est, "electricity_bill_est", externalId),
    mealIncluded: boolean(row.meal_included, "meal_included", externalId),
    foodType: required(row.food_type, "food_type", externalId),
    foodRating: decimal(row.food_rating, "food_rating", externalId, true),
    roomTypes: jsonArray(row.room_types_available, "room_types_available", externalId),
    totalRooms: integer(row.total_rooms, "total_rooms", externalId),
    totalCapacity,
    availableCapacity,
    amenities: jsonArray(row.amenities, "amenities", externalId),
    internetSpeedMbps: integer(row.internet_speed_mbps, "internet_speed_mbps", externalId),
    noiseLevel: integer(row.noise_level, "noise_level", externalId),
    curfewHour: integer(row.curfew_hour, "curfew_hour", externalId),
    studyEnvironmentScore: decimal(row.study_environment_score, "study_environment_score", externalId),
  };
}

function calculateAvailableCapacity(sourceAvailableCapacity, activeBookings, externalId) {
  const availableCapacity = sourceAvailableCapacity - activeBookings;
  if (availableCapacity < 0) {
    throw new Error(`${externalId}: active bookings exceed imported available capacity`);
  }
  return availableCapacity;
}

function valuesFor(record, availableCapacity) {
  return [
    record.externalId, record.name, record.address, record.city, record.hostelType,
    record.area, record.verified, record.yearEstablished, record.latitude, record.longitude,
    record.distanceFromFastKm, record.priceTier, record.singleRoomPrice, record.doubleRoomPrice,
    record.dormRoomPrice, record.electricityIncluded, record.electricityBillEstimate,
    record.mealIncluded, record.foodType, record.foodRating, JSON.stringify(record.roomTypes),
    record.totalRooms, record.totalCapacity, record.availableCapacity, availableCapacity, JSON.stringify(record.amenities),
    record.internetSpeedMbps, record.noiseLevel, record.curfewHour, record.studyEnvironmentScore,
  ];
}

const catalogColumns = [
  "external_id", "name", "address", "city", "hostel_type", "area", "verified",
  "year_established", "latitude", "longitude", "distance_from_fast_km", "price_tier",
  "single_room_price", "double_room_price", "dorm_room_price", "electricity_included",
  "electricity_bill_est", "meal_included", "food_type", "food_rating",
  "room_types_available", "total_rooms", "total_capacity", "source_available_capacity", "available_capacity", "amenities",
  "internet_speed_mbps", "noise_level", "curfew_hour", "study_environment_score",
];

const insertQuery = `
  INSERT INTO hostels (${catalogColumns.join(", ")}, source)
  VALUES (${catalogColumns.map((_, index) => `$${index + 1}`).join(", ")}, 'ml_catalog')
`;

const updateQuery = `
  UPDATE hostels
  SET ${catalogColumns.map((column, index) => `${column}=$${index + 1}`).join(", ")}, source='ml_catalog'
  WHERE id=$${catalogColumns.length + 1}
`;

async function main() {
  const rows = parseCsv(fs.readFileSync(sourcePath, "utf8"));
  const records = rows.map(normalizeRecord);
  const externalIds = new Set(records.map((record) => record.externalId));
  if (externalIds.size !== records.length) throw new Error("CSV contains duplicate hostel_id values");

  const client = new Client({
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  });

  await client.connect();
  let inserted = 0;
  let updated = 0;

  try {
    await client.query("BEGIN");

    for (const record of records) {
      const existing = await client.query(
        `SELECT id
         FROM hostels
         WHERE external_id=$1 OR (external_id IS NULL AND name=$2 AND city=$3)
         ORDER BY CASE WHEN external_id=$1 THEN 0 ELSE 1 END
         LIMIT 1
         FOR UPDATE`,
        [record.externalId, record.name, record.city]
      );

      const existingId = existing.rows[0]?.id;
      const activeBookings = existingId
        ? Number((await client.query(
          `SELECT COUNT(*)::int AS count
           FROM bookings
           WHERE hostel_id=$1 AND status IN ('pending', 'confirmed')`,
          [existingId]
        )).rows[0].count)
        : 0;

      const availableCapacity = calculateAvailableCapacity(
        record.availableCapacity,
        activeBookings,
        record.externalId
      );

      const values = valuesFor(record, availableCapacity);
      if (existingId) {
        await client.query(updateQuery, [...values, existingId]);
        updated += 1;
      } else {
        await client.query(insertQuery, values);
        inserted += 1;
      }
    }

    await client.query("COMMIT");
    console.log(`Catalog import complete: ${records.length} source rows, ${inserted} inserted, ${updated} updated.`);
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    await client.end();
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`Catalog import failed: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { parseCsv, normalizeRecord, calculateAvailableCapacity };