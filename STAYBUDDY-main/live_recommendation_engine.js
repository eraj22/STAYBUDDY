const ACTION_WEIGHTS = { search: 0.5, view: 1, favorite: 3, booking_attempt: 4, booking: 5 };

function clamp(value) { return Math.max(0, Math.min(1, value)); }
function normalize(map) {
  const values = [...map.values()];
  const max = Math.max(...values, 0);
  if (max <= 0) return new Map(values.length ? [...map.keys()].map((key) => [key, 0]) : []);
  return new Map([...map.entries()].map(([key, value]) => [key, value / max]));
}
function weightedJaccard(a, b) {
  const keys = new Set([...a.keys(), ...b.keys()]);
  let min = 0, max = 0;
  for (const key of keys) { min += Math.min(a.get(key) || 0, b.get(key) || 0); max += Math.max(a.get(key) || 0, b.get(key) || 0); }
  return max ? min / max : 0;
}
function addSignal(map, hostelId, weight) { map.set(Number(hostelId), (map.get(Number(hostelId)) || 0) + weight); }

async function getLiveRecommendations(pool, { studentId, topK = 5, budgetMax = null, requiredAmenities = [] }) {
  const [profiles, hostels, reviews, events] = await Promise.all([
    pool.query("SELECT user_id,university,department,budget_max,study_preference,food_preference FROM student_profiles WHERE user_id=$1", [studentId]),
    pool.query(`SELECT h.id,h.name,h.city,h.university,h.amenities,h.available_capacity,h.total_capacity,h.single_room_price,h.double_room_price,h.dorm_room_price,COALESCE(AVG(r.overall_rating),0)::float AS average_rating
      FROM hostels h LEFT JOIN reviews r ON r.hostel_id=h.id
      WHERE h.publication_status='published' AND h.available_capacity>0 GROUP BY h.id`),
    pool.query("SELECT hostel_id,AVG(overall_rating)::float AS rating FROM reviews GROUP BY hostel_id"),
    pool.query(`SELECT user_id,hostel_id,action_type,created_at FROM interactions WHERE hostel_id IS NOT NULL
      UNION ALL SELECT user_id,hostel_id,'favorite',created_at FROM favorites
      UNION ALL SELECT user_id,hostel_id,CASE WHEN status IN ('confirmed','completed') THEN 'booking' ELSE 'booking_attempt' END,created_at FROM bookings WHERE status IN ('pending','confirmed','completed')`),
  ]);
  const profile = profiles.rows[0] || {};
  const effectiveBudget = budgetMax == null ? Number(profile.budget_max) || null : Number(budgetMax);
  const amenitiesRequired = new Set(requiredAmenities.map((item) => String(item).toLowerCase()));
  const signalsByUser = new Map();
  for (const event of events.rows) {
    if (!signalsByUser.has(Number(event.user_id))) signalsByUser.set(Number(event.user_id), new Map());
    const ageDays = Math.max(0, (Date.now() - new Date(event.created_at).getTime()) / 86400000);
    addSignal(signalsByUser.get(Number(event.user_id)), event.hostel_id, (ACTION_WEIGHTS[String(event.action_type).toLowerCase()] || 0.75) * Math.exp(-0.003 * ageDays));
  }
  const mine = signalsByUser.get(Number(studentId)) || new Map();
  const peers = await pool.query("SELECT user_id,university,department FROM student_profiles WHERE user_id<>$1", [studentId]);
  const peerSimilarity = new Map();
  for (const peer of peers.rows) {
    const peerSignals = signalsByUser.get(Number(peer.user_id)) || new Map();
    const university = profile.university && peer.university && String(profile.university).toLowerCase() === String(peer.university).toLowerCase() ? 0.55 : 0;
    const department = profile.department && peer.department && String(profile.department).toLowerCase() === String(peer.department).toLowerCase() ? 0.15 : 0;
    const behaviour = weightedJaccard(mine, peerSignals) * 0.30;
    const score = university + department + behaviour;
    if (score > 0) peerSimilarity.set(Number(peer.user_id), score);
  }
  const peerAffinity = new Map();
  for (const [peerId, similarity] of peerSimilarity) {
    for (const [hostelId, weight] of signalsByUser.get(peerId) || []) addSignal(peerAffinity, hostelId, similarity * weight);
  }
  const cf = normalize(peerAffinity);
  const ratings = normalize(new Map(reviews.rows.map((row) => [Number(row.hostel_id), Number(row.rating) / 5])));
  const availability = normalize(new Map(hostels.rows.map((row) => [Number(row.id), Number(row.available_capacity)])));
  const hasCollaborativeSignal = cf.size > 0;
  const eligible = hostels.rows.filter((hostel) => {
    const prices = [hostel.single_room_price, hostel.double_room_price, hostel.dorm_room_price].filter((v) => v != null).map(Number);
    const price = prices.length ? Math.min(...prices) : null;
    const amenities = new Set((hostel.amenities || []).map((item) => String(item).toLowerCase()));
    return (!effectiveBudget || (price != null && price <= effectiveBudget)) && [...amenitiesRequired].every((item) => amenities.has(item));
  });
  const recommendations = eligible.map((hostel) => {
    const id = Number(hostel.id);
    const sameUniversity = profile.university && hostel.university && String(profile.university).toLowerCase() === String(hostel.university).toLowerCase() ? 1 : 0;
    const content = clamp(0.55 * (ratings.get(id) || 0) + 0.30 * (availability.get(id) || 0) + 0.15 * sameUniversity);
    const collaborative = cf.get(id) || 0;
    const hybrid = hasCollaborativeSignal ? 0.55 * content + 0.45 * collaborative : content;
    const reasons = [];
    if (collaborative > 0.2) reasons.push(`Preferred by students with a similar profile${profile.university ? ` at ${profile.university}` : ''}`);
    if (sameUniversity) reasons.push('Listed for your university');
    if ((ratings.get(id) || 0) > 0.7) reasons.push('Strong resident ratings');
    if ((availability.get(id) || 0) > 0.6) reasons.push('Good current availability');
    return { ...hostel, model_score: Number(hybrid.toFixed(4)), content_score: Number(content.toFixed(4)), collaborative_score: Number(collaborative.toFixed(4)), reasons: reasons.length ? reasons : ['Matches your available hostel requirements'] };
  }).filter((item) => !mine.has(Number(item.id))).sort((a, b) => b.model_score - a.model_score || b.average_rating - a.average_rating || a.id - b.id).slice(0, topK).map((item, index) => ({ rank: index + 1, ...item }));
  return { model: hasCollaborativeSignal ? 'live_hybrid_v2' : 'content_cold_start_v2', student_id: studentId, peer_count: peerSimilarity.size, interaction_count: mine.size, recommendations };
}

module.exports = { getLiveRecommendations };
