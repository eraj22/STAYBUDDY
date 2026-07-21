// recommender.js
// Hybrid recommender helpers (content + collaborative) with validation + explanations

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function softNormalize(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) return values.map(() => 0.5);
  return values.map(v => (v - min) / (max - min));
}

const ACTION_WEIGHTS = {
  search: 0.5,
  view: 1.0,
  favorite: 2.0,
  booking_attempt: 3.0,
  booking: 4.0
};

function actionWeight(actionType) {
  if (!actionType) return 0.5;
  const key = String(actionType).toLowerCase();
  return ACTION_WEIGHTS[key] ?? 0.75;
}

function buildUserPreferenceFromHistory(userInteractions, hostelMap) {
  const cityCounts = {};
  let total = 0;

  for (const it of userInteractions) {
    const h = hostelMap.get(it.hostel_id);
    if (!h) continue;
    const w = actionWeight(it.action_type);
    cityCounts[h.city] = (cityCounts[h.city] || 0) + w;
    total += w;
  }

  let topCity = null;
  let topCityScore = 0;
  for (const [city, score] of Object.entries(cityCounts)) {
    if (score > topCityScore) {
      topCityScore = score;
      topCity = city;
    }
  }

  return { topCity, cityStrength: total ? topCityScore / total : 0 };
}

function validateAndFallback(recs, options) {
  const { hasEnoughSignals } = options;

  if (!hasEnoughSignals) {
    return { usedFallback: true, reason: "insufficient_personalization_signals" };
  }

  if (recs.length >= 2 && Math.abs(recs[0].score - recs[1].score) < 0.02) {
    return { usedFallback: true, reason: "low_confidence_ranking" };
  }

  return { usedFallback: false, reason: null };
}

function explain(rec, userPref) {
  const reasons = [];
  if (rec.signals.cityMatch > 0.5 && userPref.topCity) reasons.push(`Matches your usual city: ${userPref.topCity}`);
  if (rec.signals.ratingNorm > 0.6) reasons.push("Strong review ratings");
  if (rec.signals.availNorm > 0.6) reasons.push("Good availability");
  if (rec.signals.userAffinity > 0.4) reasons.push("Similar to hostels you interacted with");

  if (reasons.length === 0) reasons.push("Recommended based on overall compatibility signals");
  return reasons.slice(0, 3);
}

module.exports = {
  buildUserPreferenceFromHistory,
  softNormalize,
  validateAndFallback,
  explain,
  actionWeight,
  clamp01
};
