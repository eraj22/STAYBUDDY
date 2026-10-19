const fs = require("node:fs");
const path = require("node:path");

// A lightweight local ML fallback for computers where the separate Python
// service cannot start.  It trains from complaints.csv at backend startup;
// it is not a hard-coded keyword list or a demo response.
const DATA_PATH = path.join(__dirname, "ml-notebooks", "ml_ai_complaint", "complaints.csv");

function parseCsvRow(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += character;
    }
  }
  values.push(value);
  return values;
}

function readComplaintRows() {
  const lines = fs.readFileSync(DATA_PATH, "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
  const headers = parseCsvRow(lines.shift());
  return lines.map((line) => {
    const values = parseCsvRow(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] || ""]));
  }).filter((row) => row.complaint_text && row.category);
}

function normalize(text) {
  let value = String(text || "").toLowerCase();
  const expansions = {
    wifi: "wifi internet connection",
    internet: "internet wifi connection",
    warden: "warden staff management",
    cctv: "cctv camera security surveillance",
    stolen: "stolen theft missing",
    cockroach: "cockroach pest insect",
    smell: "smell odor dirty unhygienic",
    ac: "air conditioning cooling",
  };
  for (const [term, expansion] of Object.entries(expansions)) {
    value = value.replace(new RegExp(`\\b${term}\\b`, "g"), expansion);
  }
  return value.replace(/[^a-z\s]/g, " ").replace(/\s+/g, " ").trim();
}

function tokens(text) {
  return normalize(text).split(" ").filter((token) => token.length > 1);
}

function buildModel() {
  const rows = readComplaintRows();
  const categories = new Map();
  const vocabulary = new Set();

  for (const row of rows) {
    const category = row.category;
    if (!categories.has(category)) {
      categories.set(category, { documents: 0, tokenCount: 0, tokens: new Map() });
    }
    const bucket = categories.get(category);
    bucket.documents += 1;
    for (const token of tokens(row.complaint_text)) {
      vocabulary.add(token);
      bucket.tokenCount += 1;
      bucket.tokens.set(token, (bucket.tokens.get(token) || 0) + 1);
    }
  }
  return { rows, categories, vocabularySize: Math.max(vocabulary.size, 1) };
}

let model;
try {
  model = buildModel();
  console.log(`Complaint CSV model ready (${model.rows.length} training examples).`);
} catch (error) {
  console.warn("Complaint CSV model unavailable:", error.message);
  model = null;
}

function cosineSimilarity(leftTokens, rightTokens) {
  const left = new Map();
  const right = new Map();
  for (const token of leftTokens) left.set(token, (left.get(token) || 0) + 1);
  for (const token of rightTokens) right.set(token, (right.get(token) || 0) + 1);
  let dot = 0;
  for (const [token, count] of left) dot += count * (right.get(token) || 0);
  const leftMagnitude = Math.sqrt([...left.values()].reduce((sum, count) => sum + count * count, 0));
  const rightMagnitude = Math.sqrt([...right.values()].reduce((sum, count) => sum + count * count, 0));
  return leftMagnitude && rightMagnitude ? dot / (leftMagnitude * rightMagnitude) : 0;
}

function analyzeCsvComplaint(description) {
  if (!model || !String(description || "").trim()) return null;
  const inputTokens = tokens(description);
  if (!inputTokens.length) return null;

  const totalDocuments = model.rows.length;
  const scores = [];
  for (const [category, bucket] of model.categories) {
    let score = Math.log(bucket.documents / totalDocuments);
    for (const token of inputTokens) {
      score += Math.log(((bucket.tokens.get(token) || 0) + 1) / (bucket.tokenCount + model.vocabularySize));
    }
    scores.push({ category, score });
  }
  scores.sort((left, right) => right.score - left.score);
  const maxScore = scores[0].score;
  const denominator = scores.reduce((sum, item) => sum + Math.exp(item.score - maxScore), 0);
  const top3 = scores.slice(0, 3).map((item) => ({
    category: item.category,
    confidence: Number((Math.exp(item.score - maxScore) / denominator).toFixed(3)),
  }));
  const category = top3[0].category;
  const examples = model.rows.filter((row) => row.category === category);
  const closest = examples.reduce((best, row) => {
    const similarity = cosineSimilarity(inputTokens, tokens(row.complaint_text));
    return !best || similarity > best.similarity ? { row, similarity } : best;
  }, null);
  const suggestions = [...new Set(examples.map((row) => row.ai_suggestion).filter(Boolean))].slice(0, 5);

  return {
    success: true,
    model_version: "complaints.csv Naive Bayes + nearest-example v1",
    category,
    subcategory: closest?.row.subcategory || "General",
    priority: closest?.row.priority || "Medium",
    confidence: top3[0].confidence,
    suggestion: closest?.row.ai_suggestion || suggestions[0] || "Review the complaint and respond within 48 hours.",
    all_suggestions: suggestions,
    top3_categories: top3,
  };
}

module.exports = { analyzeCsvComplaint };
