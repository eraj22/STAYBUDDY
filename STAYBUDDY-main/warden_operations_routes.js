function registerWardenOperationsRoutes({ app, pool, requireAuth, requireRole, isValidInt }) {
  // No seed records: a new hostel starts with genuine empty operational lists.
  const tablesReady = pool.query(`
    CREATE TABLE IF NOT EXISTS warden_penalties (
      id SERIAL PRIMARY KEY,
      hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
      student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      reason VARCHAR(240) NOT NULL,
      amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
      note TEXT,
      status VARCHAR(12) NOT NULL DEFAULT 'unpaid' CHECK (status IN ('unpaid','paid')),
      created_by_warden_id INTEGER NOT NULL REFERENCES users(id),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      paid_at TIMESTAMPTZ,
      paid_by_warden_id INTEGER REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS cleaning_schedules (
      id SERIAL PRIMARY KEY,
      hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
      area VARCHAR(300) NOT NULL,
      scheduled_date DATE NOT NULL,
      shift VARCHAR(80) NOT NULL,
      status VARCHAR(12) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','completed')),
      created_by_warden_id INTEGER NOT NULL REFERENCES users(id),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      completed_at TIMESTAMPTZ,
      completed_by_warden_id INTEGER REFERENCES users(id)
    );
  `);
  const hostelFor = (userId) => pool.query('SELECT hostel_id FROM warden_assignments WHERE warden_user_id=$1 LIMIT 1', [userId]);

  app.get('/api/warden/penalties', requireAuth, requireRole('warden'), async (req, res) => {
    try {
      await tablesReady;
      const result = await pool.query(`SELECT p.*,u.name AS student_name,a.floor_label,a.room_number,a.bed_number
        FROM warden_penalties p JOIN users u ON u.id=p.student_user_id
        JOIN warden_assignments wa ON wa.hostel_id=p.hostel_id AND wa.warden_user_id=$1
        LEFT JOIN LATERAL(SELECT COALESCE(r.floor_label,'Ground Floor') floor_label,r.room_number,b.bed_number FROM bookings b LEFT JOIN rooms r ON r.id=b.room_id WHERE b.hostel_id=p.hostel_id AND b.user_id=p.student_user_id ORDER BY b.updated_at DESC,b.id DESC LIMIT 1)a ON true
        ORDER BY p.created_at DESC,p.id DESC`, [req.user.id]);
      res.json(result.rows);
    } catch (_) { res.status(500).json({ error: 'Failed to fetch penalties' }); }
  });

  app.post('/api/warden/penalties', requireAuth, requireRole('warden'), async (req, res) => {
    try {
      await tablesReady;
      const studentId = Number(req.body?.student_id), reason = String(req.body?.reason || '').trim(), amount = Number(req.body?.amount), note = String(req.body?.note || '').trim() || null;
      if (!isValidInt(studentId) || !reason || reason.length > 240 || !Number.isFinite(amount) || amount <= 0) return res.status(400).json({ error: 'Student, reason, and valid amount are required' });
      const hostel = await hostelFor(req.user.id);
      if (!hostel.rowCount) return res.status(404).json({ error: 'Assigned hostel not found' });
      const resident = await pool.query("SELECT 1 FROM bookings WHERE hostel_id=$1 AND user_id=$2 AND status='confirmed'", [hostel.rows[0].hostel_id, studentId]);
      if (!resident.rowCount) return res.status(409).json({ error: 'Student is not a confirmed resident' });
      const result = await pool.query('INSERT INTO warden_penalties(hostel_id,student_user_id,reason,amount,note,created_by_warden_id) VALUES($1,$2,$3,$4,$5,$6) RETURNING *', [hostel.rows[0].hostel_id, studentId, reason, amount, note, req.user.id]);
      res.status(201).json({ penalty: result.rows[0] });
    } catch (_) { res.status(500).json({ error: 'Failed to add penalty' }); }
  });

  app.patch('/api/warden/penalties/:id/paid', requireAuth, requireRole('warden'), async (req, res) => {
    try {
      await tablesReady;
      const id = Number(req.params.id); if (!isValidInt(id)) return res.status(400).json({ error: 'Invalid penalty' });
      const result = await pool.query(`UPDATE warden_penalties p SET status='paid',paid_at=NOW(),paid_by_warden_id=$2 WHERE p.id=$1 AND EXISTS(SELECT 1 FROM warden_assignments wa WHERE wa.hostel_id=p.hostel_id AND wa.warden_user_id=$2) RETURNING *`, [id, req.user.id]);
      if (!result.rowCount) return res.status(404).json({ error: 'Penalty not found' });
      res.json({ penalty: result.rows[0] });
    } catch (_) { res.status(500).json({ error: 'Failed to mark penalty paid' }); }
  });

  app.get('/api/warden/cleaning-schedule', requireAuth, requireRole('warden'), async (req, res) => {
    try {
      await tablesReady;
      const result = await pool.query(`SELECT c.* FROM cleaning_schedules c JOIN warden_assignments wa ON wa.hostel_id=c.hostel_id WHERE wa.warden_user_id=$1 ORDER BY c.scheduled_date ASC,c.id DESC`, [req.user.id]);
      res.json(result.rows);
    } catch (_) { res.status(500).json({ error: 'Failed to fetch cleaning schedule' }); }
  });

  app.post('/api/warden/cleaning-schedule', requireAuth, requireRole('warden'), async (req, res) => {
    try {
      await tablesReady;
      const area = String(req.body?.area || '').trim(), date = String(req.body?.scheduled_date || '').trim(), shift = String(req.body?.shift || '').trim();
      if (!area || area.length > 300 || !/^\d{4}-\d{2}-\d{2}$/.test(date) || !shift || shift.length > 80) return res.status(400).json({ error: 'Area, date, and shift are required' });
      const hostel = await hostelFor(req.user.id);
      if (!hostel.rowCount) return res.status(404).json({ error: 'Assigned hostel not found' });
      const result = await pool.query('INSERT INTO cleaning_schedules(hostel_id,area,scheduled_date,shift,created_by_warden_id) VALUES($1,$2,$3::date,$4,$5) RETURNING *', [hostel.rows[0].hostel_id, area, date, shift, req.user.id]);
      res.status(201).json({ schedule: result.rows[0] });
    } catch (_) { res.status(500).json({ error: 'Failed to create cleaning schedule' }); }
  });

  app.patch('/api/warden/cleaning-schedule/:id', requireAuth, requireRole('warden'), async (req, res) => {
    try {
      await tablesReady;
      const id = Number(req.params.id), status = String(req.body?.status || '');
      if (!isValidInt(id) || !['pending', 'completed'].includes(status)) return res.status(400).json({ error: 'Valid schedule and status are required' });
      const result = await pool.query(`UPDATE cleaning_schedules c SET status=$2,completed_at=CASE WHEN $2='completed' THEN NOW() ELSE NULL END,completed_by_warden_id=CASE WHEN $2='completed' THEN $3 ELSE NULL END WHERE c.id=$1 AND EXISTS(SELECT 1 FROM warden_assignments wa WHERE wa.hostel_id=c.hostel_id AND wa.warden_user_id=$3) RETURNING *`, [id, status, req.user.id]);
      if (!result.rowCount) return res.status(404).json({ error: 'Cleaning schedule not found' });
      res.json({ schedule: result.rows[0] });
    } catch (_) { res.status(500).json({ error: 'Failed to update cleaning schedule' }); }
  });
}

module.exports = { registerWardenOperationsRoutes };
