// Adds idempotent, fictional verified hostels for live recommender testing.
// Run inside the api container: node scripts/seed_recommendation_demo_hostels.js
const { Pool } = require("pg");
const { hashPassword } = require("../passwords");

const hostels = [
  ["FAST", "FAST Girls Residence G-11", "Girls", "G-11", 33.676, 72.999, 16500], ["FAST", "CodeNest Boys Hostel", "Boys", "H-13", 33.645, 72.992, 14500], ["FAST", "ByteHouse Student Living", "Girls", "G-10", 33.681, 73.008, 18500],
  ["NUST", "NUST Gate Residence", "Girls", "H-12", 33.642, 72.989, 17500], ["NUST", "H-13 Scholars Hostel", "Boys", "H-13", 33.650, 72.996, 15000], ["NUST", "Capital Campus Lodge", "Girls", "H-11", 33.660, 72.985, 19000],
  ["COMSATS University Islamabad", "COMSATS Comfort Hostel", "Girls", "Park Road", 33.616, 73.121, 16000], ["COMSATS University Islamabad", "Lake View Boys Residence", "Boys", "Tarlai", 33.601, 73.132, 13500], ["COMSATS University Islamabad", "Park Road Student House", "Girls", "Chak Shahzad", 33.626, 73.115, 18000],
  ["Air University Islamabad", "Air Campus Residence", "Girls", "E-9", 33.738, 73.081, 19000], ["Air University Islamabad", "Margalla Boys Hostel", "Boys", "E-11", 33.704, 72.979, 17000],
  ["Bahria University Islamabad", "Bahria Student Residence", "Girls", "E-8", 33.732, 73.094, 18500], ["Bahria University Islamabad", "Naval Heights Hostel", "Boys", "F-8", 33.718, 73.081, 17500],
  ["Quaid-i-Azam University", "QAU Valley Residence", "Girls", "Bhara Kahu", 33.751, 73.146, 14000], ["Quaid-i-Azam University", "University View Boys Hostel", "Boys", "Bhara Kahu", 33.747, 73.151, 13000],
];

async function main() {
  const pool = new Pool({ host: process.env.DB_HOST, port: Number(process.env.DB_PORT || 5432), user: process.env.DB_USER, password: process.env.DB_PASSWORD, database: process.env.DB_NAME });
  try {
    for (let index = 0; index < hostels.length; index += 1) {
      const [university, name, hostelType, area, latitude, longitude, price] = hostels[index];
      await pool.query(`INSERT INTO hostels (external_id,name,address,city,university,hostel_type,area,latitude,longitude,total_capacity,available_capacity,source_available_capacity,description,verified,publication_status,amenities,facilities,rules,mess_info,media,room_types_available,single_room_price,double_room_price,total_rooms,source)
        VALUES ($1,$2,$3,'Islamabad',$4,$5,$6,$7,$8,30,12,12,$9,true,'published',$10::jsonb,$11::jsonb,$12::jsonb,$13::jsonb,'[]'::jsonb,$14::jsonb,$15,$16,15,'recommendation_demo')
        ON CONFLICT (external_id) WHERE external_id IS NOT NULL DO UPDATE SET name=EXCLUDED.name,university=EXCLUDED.university,verified=true,publication_status='published',available_capacity=EXCLUDED.available_capacity,amenities=EXCLUDED.amenities,single_room_price=EXCLUDED.single_room_price,double_room_price=EXCLUDED.double_room_price,source='recommendation_demo'`,
        [`RECOMMENDER-DEMO-${String(index + 1).padStart(2, "0")}`, name, `Fictional address, ${area}, Islamabad`, university, hostelType, area, latitude, longitude, `Fictional verified demo hostel for ${university} students.`, JSON.stringify(["wifi", "security guard", "laundry", index % 2 ? "hot water" : "study room"]), JSON.stringify(["generator", "common room"]), JSON.stringify(["Quiet hours after 10 PM"]), JSON.stringify({ available: true }), JSON.stringify(["single", "double"]), price, price - 3500]);
    }

    // A repeatable peer signal makes it possible to demonstrate collaborative
    // recommendations immediately. A newly registered NUST student should see
    // the NUST Gate Residence with a peer-based explanation.
    const passwordHash = await hashPassword("DemoStudent123!");
    const peer = await pool.query(
      `INSERT INTO users (name,email,password_hash,role,phone,account_status)
       VALUES ('NUST Demo Peer','nust.peer@staybuddy.test',$1,'student','0300-0000099','active')
       ON CONFLICT (email) DO UPDATE SET password_hash=EXCLUDED.password_hash,account_status='active'
       RETURNING id`,
      [passwordHash]
    );
    await pool.query(
      `INSERT INTO student_profiles (user_id,university,department,gender,budget_max,study_preference)
       VALUES ($1,'NUST','Computer Science','Female',22000,0.6)
       ON CONFLICT (user_id) DO UPDATE SET university='NUST',department='Computer Science',budget_max=22000,study_preference=0.6,updated_at=NOW()`,
      [peer.rows[0].id]
    );
    const peerHostel = await pool.query("SELECT id FROM hostels WHERE external_id='RECOMMENDER-DEMO-04'");
    await pool.query(
      "INSERT INTO favorites (user_id,hostel_id) VALUES ($1,$2) ON CONFLICT (user_id,hostel_id) DO NOTHING",
      [peer.rows[0].id, peerHostel.rows[0].id]
    );
    console.log(`Seeded ${hostels.length} verified recommendation-demo hostels and one NUST peer preference.`);
  } finally { await pool.end(); }
}
main().catch((error) => { console.error(error); process.exit(1); });
