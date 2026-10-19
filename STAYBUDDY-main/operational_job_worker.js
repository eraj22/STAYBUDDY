const crypto = require("crypto");
const { sendExternalDelivery } = require("./notification_providers");
const { enqueueOperationalJob } = require("./job_queue");
const { currentBillingMonth, generateFeeInvoices } = require("./fee_invoice_generator");

function retryAt(attemptNumber) {
  const minutes = Math.min(60, 2 ** Math.max(0, attemptNumber - 1));
  return new Date(Date.now() + minutes * 60 * 1000);
}

async function processDelivery(client, job, table, idColumn) {
  const deliveryId = Number(job.payload.delivery_id);
  const result = await client.query(
    table === "emergency_delivery_attempts"
      ? `SELECT d.*, ec.name AS contact_name, ec.phone AS contact_phone, ec.email AS contact_email, u.name AS user_name, u.email AS user_email, pt.token AS push_token,
          e.alert_type, e.description FROM emergency_delivery_attempts d
          JOIN emergency_incidents e ON e.id=d.incident_id
          LEFT JOIN emergency_contacts ec ON ec.id=d.contact_id LEFT JOIN users u ON u.id=d.recipient_user_id
          LEFT JOIN LATERAL (SELECT token FROM user_push_tokens WHERE user_id=u.id AND active=true ORDER BY updated_at DESC LIMIT 1) pt ON true
          WHERE d.id=$1 FOR UPDATE OF d`
      : `SELECT d.*, u.name AS user_name, u.email AS user_email, pt.token AS push_token, i.amount, i.currency, i.due_date FROM fee_reminder_deliveries d
          JOIN users u ON u.id=d.recipient_user_id JOIN fee_invoices i ON i.id=d.invoice_id
          LEFT JOIN LATERAL (SELECT token FROM user_push_tokens WHERE user_id=u.id AND active=true ORDER BY updated_at DESC LIMIT 1) pt ON true
          WHERE d.id=$1 FOR UPDATE OF d`,
    [deliveryId]
  );
  if (!result.rowCount) return { delivered: true, detail: "delivery no longer exists" };
  const delivery = result.rows[0];
  if (delivery.status === "delivered") return { delivered: true, detail: "delivery already completed" };
  const recipient = { name: delivery.contact_name || delivery.user_name, phone: delivery.contact_phone || null, email: delivery.contact_email || delivery.user_email || null, push_token: delivery.push_token || null };
  const subject = table === "emergency_delivery_attempts" ? "StayBuddy emergency alert" : "StayBuddy fee payment reminder";
  const body = table === "emergency_delivery_attempts"
    ? `Emergency alert: ${delivery.alert_type}. ${delivery.description || "Please check StayBuddy immediately."}`
    : `A fee invoice of ${delivery.amount} ${delivery.currency} is due on ${delivery.due_date}.`;
  const outcome = await sendExternalDelivery({ channel: delivery.channel, recipient, subject, body });
  if (outcome.delivered) {
    await client.query(`UPDATE ${table} SET status='delivered', provider=$1, provider_reference=$2, attempt_count=attempt_count+1, error_message=NULL, next_retry_at=NULL, updated_at=NOW() WHERE ${idColumn}=$3`, [outcome.provider, outcome.providerReference || null, deliveryId]);
    return { delivered: true, provider: outcome.provider };
  }
  await client.query(`UPDATE ${table} SET status='failed', provider=$1, attempt_count=attempt_count+1, error_message=$2, updated_at=NOW() WHERE ${idColumn}=$3`, [outcome.provider || null, outcome.error || "Delivery failed", deliveryId]);
  return { delivered: false, error: outcome.error || "Delivery failed" };
}

async function processConditionDeadlineCheck(client) {
  const overdue = await client.query("UPDATE hostel_approval_conditions SET status='overdue' WHERE status='open' AND due_at<NOW() RETURNING id,application_id");
  const expired = await client.query("UPDATE hostel_verification_information_requests SET status='expired' WHERE status='open' AND due_at<NOW() RETURNING id,application_id");
  return { delivered: true, detail: `${overdue.rowCount} condition(s) marked overdue; ${expired.rowCount} information request(s) expired` };
}

async function processInvoiceOverdueCheck(client) {
  const overdue = await client.query("UPDATE fee_invoices SET status='overdue',updated_at=NOW() WHERE status='pending' AND due_date<CURRENT_DATE RETURNING id");
  return { delivered: true, detail: `${overdue.rowCount} invoice(s) marked overdue` };
}

async function processFeeInvoiceGeneration(client) {
  const result = await generateFeeInvoices(client, { billingMonth: currentBillingMonth() });
  return { delivered: true, detail: `${result.created} invoice(s) generated for ${result.billingMonth}`, ...result };
}

async function processFeeReminderCheck(client) {
  const invoices = await client.query(`SELECT i.id,i.amount,i.currency,i.due_date,a.student_user_id,s.hostel_id,(CURRENT_DATE-i.due_date)::int AS days_from_due
    FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id
    WHERE i.status IN ('pending','overdue') AND (CURRENT_DATE-i.due_date) IN (-7,-5,-3,-1,0,1,3,7)`);
  let sent = 0;
  for (const invoice of invoices.rows) {
    const code = `${invoice.days_from_due >= 0 ? 'overdue' : 'before'}_${Math.abs(invoice.days_from_due)}_days`;
    const event = await client.query("INSERT INTO fee_invoice_reminder_events (invoice_id,reminder_code) VALUES ($1,$2) ON CONFLICT DO NOTHING RETURNING id", [invoice.id, code]);
    if (!event.rowCount) continue;
    const phase = invoice.days_from_due > 0 ? `is overdue by ${invoice.days_from_due} day(s)` : invoice.days_from_due === 0 ? "is due today" : `is due in ${Math.abs(invoice.days_from_due)} day(s)`;
    const recipients = await client.query("SELECT $1::int AS user_id UNION SELECT parent_user_id FROM parent_student_links WHERE student_user_id=$1 AND status='active' UNION SELECT warden_user_id FROM warden_assignments WHERE hostel_id=$2", [invoice.student_user_id, invoice.hostel_id]);
    for (const recipient of recipients.rows) await client.query("INSERT INTO notifications (user_id,type,title,body,data) SELECT $1,'fee_reminder','Fee payment reminder',$2,$3::jsonb WHERE COALESCE((SELECT fee_reminders FROM notification_preferences WHERE user_id=$1),true)", [recipient.user_id, `A PKR ${invoice.amount} fee notice ${phase}.`, JSON.stringify({ invoice_id: invoice.id, due_date: invoice.due_date, reminder_code: code })]);
    sent += recipients.rowCount;
  }
  return { delivered: true, detail: `${sent} fee reminder notification(s) created`, sent };
}

async function processComplaintSlaCheck(client, workerId) {
  const overdue = await client.query("UPDATE complaints SET escalated_at=NOW() WHERE sla_due_at<NOW() AND escalated_at IS NULL AND status NOT IN ('resolved','closed') RETURNING id,hostel_id,priority,sla_due_at");
  for (const complaint of overdue.rows) {
    await client.query("INSERT INTO complaint_sla_escalations (complaint_id,escalated_by_worker,details) VALUES ($1,$2,$3::jsonb) ON CONFLICT (complaint_id) DO NOTHING", [complaint.id,workerId,JSON.stringify({priority:complaint.priority,sla_due_at:complaint.sla_due_at})]);
    await client.query("INSERT INTO notifications (user_id,type,title,body,data) SELECT ho.owner_user_id,'complaint_sla_escalated','Complaint SLA overdue','A complaint has exceeded its SLA deadline.',jsonb_build_object('complaint_id',$1::integer) FROM hostel_owners ho JOIN notification_preferences np ON np.user_id=ho.owner_user_id WHERE ho.hostel_id=$2::integer AND np.complaint_updates=true",[complaint.id,complaint.hostel_id]);
  }
  return { delivered:true, detail:`${overdue.rowCount} complaint(s) escalated` };
}

async function processComplaintVolumeCheck(client, workerId) {
  const rules = await client.query("SELECT * FROM hostel_complaint_alert_rules WHERE enabled=true"); let triggered = 0;
  for (const rule of rules.rows) {
    const count = await client.query("SELECT count(*)::int AS count FROM complaints WHERE hostel_id=$1 AND status NOT IN ('resolved','closed')", [rule.hostel_id]); const openCount = count.rows[0].count;
    if (openCount < rule.open_complaint_threshold) continue;
    const recent = await client.query("SELECT 1 FROM complaint_volume_alerts WHERE hostel_id=$1 AND triggered_at > NOW() - ($2::int * INTERVAL '1 minute') LIMIT 1", [rule.hostel_id, rule.cooldown_minutes]);
    if (recent.rowCount) continue;
    const alert = await client.query("INSERT INTO complaint_volume_alerts (hostel_id,threshold_count,observed_open_count,triggered_by_worker) VALUES ($1,$2,$3,$4) RETURNING *", [rule.hostel_id, rule.open_complaint_threshold, openCount, workerId]);
    const recipients = await client.query("SELECT owner_user_id AS user_id FROM hostel_owners WHERE hostel_id=$1 UNION SELECT warden_user_id FROM warden_assignments WHERE hostel_id=$1", [rule.hostel_id]);
    for (const recipient of recipients.rows) await client.query("INSERT INTO notifications (user_id,type,title,body,data) SELECT $1,'complaint_volume_alert','Complaint volume threshold reached',$2,$3::jsonb WHERE COALESCE((SELECT complaint_updates FROM notification_preferences WHERE user_id=$1),true)", [recipient.user_id, `There are ${openCount} open complaints at this hostel.`, JSON.stringify({ hostel_id: rule.hostel_id, complaint_volume_alert_id: alert.rows[0].id, open_complaint_count: openCount })]);
    triggered += 1;
  }
  return { delivered: true, detail: `${triggered} complaint volume alert(s) triggered`, triggered };
}

async function processLocationRetentionCheck(client) {
  const expired = await client.query("UPDATE location_sharing_sessions SET revoked_at=NOW() WHERE revoked_at IS NULL AND expires_at<=NOW() RETURNING id,student_user_id");
  for (const row of expired.rows) await client.query("INSERT INTO location_sharing_audit_events (session_id,event_type,details) VALUES ($1,'expired',$2::jsonb)", [row.id, JSON.stringify({ student_user_id: row.student_user_id })]);
  const removed = await client.query("DELETE FROM location_sharing_updates WHERE reported_at < NOW()-INTERVAL '7 days'");
  const gateRedacted = await client.query(`UPDATE attendance_gate_requests
    SET latitude=NULL,longitude=NULL,accuracy_m=NULL
    WHERE submitted_at<NOW()-INTERVAL '7 days' AND latitude IS NOT NULL`);
  return { delivered: true, detail: `${expired.rowCount} session(s) expired; ${removed.rowCount} location update(s) deleted; ${gateRedacted.rowCount} gate coordinate(s) redacted` };
}

async function processAvailabilityAlertCheck(client) {
  const alerts = await client.query("SELECT * FROM hostel_availability_alerts WHERE active=true");
  let matched = 0;
  for (const alert of alerts.rows) {
    const params = [];
    const add = (value) => { params.push(value); return `$${params.length}`; };
    const conditions = ["h.verified=true", "h.publication_status='published'", "h.available_capacity>0"];
    if (alert.university) conditions.push(`h.university ILIKE ${add(`%${alert.university}%`)}`);
    if (alert.city) conditions.push(`h.city ILIKE ${add(`%${alert.city}%`)}`);
    if (alert.hostel_type) conditions.push(`h.hostel_type ILIKE ${add(`%${alert.hostel_type}%`)}`);
    if (alert.max_price != null) conditions.push(`(SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id=h.id AND available_capacity>0) <= ${add(Number(alert.max_price))}`);
    if (alert.latitude != null) {
      const latitude = add(Number(alert.latitude)), longitude = add(Number(alert.longitude)), radius = add(Number(alert.radius_km));
      conditions.push("h.latitude IS NOT NULL AND h.longitude IS NOT NULL");
      conditions.push(`(6371 * acos(LEAST(1,GREATEST(-1,cos(radians(${latitude})) * cos(radians(h.latitude)) * cos(radians(h.longitude)-radians(${longitude})) + sin(radians(${latitude})) * sin(radians(h.latitude)))))) <= ${radius}`);
    }
    const hostels = await client.query(`SELECT h.id,h.name FROM hostels h WHERE ${conditions.join(" AND ")}`, params);
    for (const hostel of hostels.rows) {
      const inserted = await client.query("INSERT INTO hostel_availability_alert_matches (alert_id,hostel_id) VALUES ($1,$2) ON CONFLICT DO NOTHING RETURNING *", [alert.id, hostel.id]);
      if (!inserted.rowCount) continue;
      await client.query("INSERT INTO notifications (user_id,type,title,body,data) SELECT $1,'hostel_availability','Hostel now available',$2,$3::jsonb WHERE COALESCE((SELECT availability_updates FROM notification_preferences WHERE user_id=$1),true)", [alert.student_user_id, `${hostel.name} matches your saved availability alert.`, JSON.stringify({ alert_id: alert.id, hostel_id: hostel.id })]);
      matched += 1;
    }
  }
  const favorites = await client.query(`SELECT f.id,f.user_id,f.hostel_id,f.price_snapshot,f.availability_snapshot,
      h.name,h.available_capacity,
      (SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id=h.id AND available_capacity>0) AS display_price
    FROM favorites f JOIN hostels h ON h.id=f.hostel_id
    WHERE h.publication_status='published'`);
  let favoriteChanges = 0;
  for (const favorite of favorites.rows) {
    const previousPrice = favorite.price_snapshot == null ? null : Number(favorite.price_snapshot);
    const currentPrice = favorite.display_price == null ? null : Number(favorite.display_price);
    const previousAvailability = favorite.availability_snapshot == null ? null : Number(favorite.availability_snapshot);
    const currentAvailability = Number(favorite.available_capacity || 0);
    if (previousPrice != null && currentPrice != null && currentPrice < previousPrice) {
      await client.query("INSERT INTO notifications (user_id,type,title,body,data) SELECT $1,'favorite_price_drop','Saved hostel price dropped',$2,$3::jsonb WHERE COALESCE((SELECT availability_updates FROM notification_preferences WHERE user_id=$1),true)", [favorite.user_id, `${favorite.name} now starts at PKR ${currentPrice}, down from PKR ${previousPrice}.`, JSON.stringify({ favorite_id: favorite.id, hostel_id: favorite.hostel_id, previous_price: previousPrice, current_price: currentPrice })]);
      favoriteChanges += 1;
    }
    if (previousAvailability != null && previousAvailability <= 0 && currentAvailability > 0) {
      await client.query("INSERT INTO notifications (user_id,type,title,body,data) SELECT $1,'favorite_available','Saved hostel is available',$2,$3::jsonb WHERE COALESCE((SELECT availability_updates FROM notification_preferences WHERE user_id=$1),true)", [favorite.user_id, `${favorite.name} has availability again.`, JSON.stringify({ favorite_id: favorite.id, hostel_id: favorite.hostel_id, previous_availability: previousAvailability, current_availability: currentAvailability })]);
      favoriteChanges += 1;
    }
    if (previousPrice !== currentPrice || previousAvailability !== currentAvailability) {
      await client.query("UPDATE favorites SET price_snapshot=$1,availability_snapshot=$2,updated_at=NOW() WHERE id=$3", [currentPrice, currentAvailability, favorite.id]);
    }
  }
  return { delivered: true, detail: `${matched} new availability match(es) and ${favoriteChanges} favorite change(s) notified`, matched, favorite_changes: favoriteChanges };
}

async function processBookingExpiryCheck(client) {
  const expired=await client.query("UPDATE bookings SET status='expired',updated_at=NOW() WHERE status='pending' AND expires_at<=NOW() RETURNING id,hostel_id,room_id");
  for(const booking of expired.rows){await client.query("UPDATE hostels SET available_capacity=LEAST(total_capacity,available_capacity+1) WHERE id=$1",[booking.hostel_id]);const hold=await client.query("UPDATE booking_inventory_holds SET released_at=NOW() WHERE booking_id=$1 AND released_at IS NULL AND converted_at IS NULL RETURNING room_id",[booking.id]);if(hold.rowCount)await client.query("UPDATE rooms SET available_capacity=LEAST(capacity,available_capacity+1) WHERE id=$1",[hold.rows[0].room_id]);else if(booking.room_id)await client.query("UPDATE rooms SET available_capacity=LEAST(capacity,available_capacity+1) WHERE id=$1",[booking.room_id]);await client.query("INSERT INTO booking_status_history(booking_id,previous_status,next_status,reason) VALUES ($1,'pending','expired','24-hour reservation hold expired')",[booking.id]);}
  return {delivered:true,detail:`${expired.rowCount} pending booking(s) expired`};
}

async function processComplaintAutoClose(client) {
  const closed=await client.query("UPDATE complaints SET status='closed',resolution_due_at=NULL WHERE status='resolved' AND resolution_due_at<=NOW() RETURNING id");
  for(const complaint of closed.rows)await client.query("INSERT INTO complaint_status_history(complaint_id,previous_status,next_status,changed_by_user_id) VALUES ($1,'resolved','closed',NULL)",[complaint.id]);
  return {delivered:true,detail:`${closed.rowCount} unchallenged complaint(s) closed`};
}

async function processAttendanceCurfewCheck(client) {
  const settings = await client.query(`SELECT hostel_id,timezone,escalation_minutes,
    (((NOW() AT TIME ZONE timezone)::date + curfew_time) AT TIME ZONE timezone) AS curfew_at
    FROM hostel_attendance_settings WHERE enabled=true`);
  const now = new Date();
  let created = 0, notified = 0;
  for (const setting of settings.rows) {
    if (now < new Date(setting.curfew_at)) continue;
    const students = await client.query(`SELECT b.user_id,MAX(s.checked_in_at) AS checked_in_at FROM bookings b JOIN booking_stays s ON s.booking_id=b.id
      WHERE b.hostel_id=$1 AND b.status='confirmed' AND s.checked_in_at IS NOT NULL AND s.checked_out_at IS NULL
      GROUP BY b.user_id`, [setting.hostel_id]);
    for (const student of students.rows) {
      const latest = await client.query(`SELECT e.id,COALESCE(c.corrected_direction,e.direction) AS direction
        FROM attendance_events e LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac
          WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
        WHERE e.hostel_id=$1 AND e.student_user_id=$2 AND COALESCE(c.corrected_occurred_at,e.occurred_at)<=$3
          AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$4
        ORDER BY COALESCE(c.corrected_occurred_at,e.occurred_at) DESC,e.id DESC LIMIT 1`, [setting.hostel_id, student.user_id, setting.curfew_at, student.checked_in_at]);
      if (!latest.rowCount || latest.rows[0].direction !== 'out') continue;
      const current = await client.query(`SELECT COALESCE(c.corrected_direction,e.direction) AS direction
        FROM attendance_events e LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac
          WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
        WHERE e.hostel_id=$1 AND e.student_user_id=$2 AND COALESCE(c.corrected_occurred_at,e.occurred_at)<=$3
          AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$4
        ORDER BY COALESCE(c.corrected_occurred_at,e.occurred_at) DESC,e.id DESC LIMIT 1`, [setting.hostel_id, student.user_id, now, student.checked_in_at]);
      if (!current.rowCount || current.rows[0].direction !== 'out') continue;
      const inserted = await client.query(`INSERT INTO attendance_exceptions(hostel_id,student_user_id,policy_date,exception_type,details)
        VALUES ($1,$2,($3::timestamptz AT TIME ZONE $4)::date,'absent_at_curfew',$5::jsonb)
        ON CONFLICT DO NOTHING RETURNING id`, [setting.hostel_id, student.user_id, setting.curfew_at, setting.timezone,
        JSON.stringify({last_attendance_event_id:latest.rows[0].id,durable_job:true})]);
      const existing = inserted.rowCount ? inserted : await client.query(`SELECT id FROM attendance_exceptions
        WHERE hostel_id=$1 AND student_user_id=$2 AND policy_date=($3::timestamptz AT TIME ZONE $4)::date
          AND exception_type='absent_at_curfew'`, [setting.hostel_id, student.user_id, setting.curfew_at, setting.timezone]);
      if (!existing.rowCount) continue;
      const exceptionId = existing.rows[0].id;
      const parents = await client.query(`SELECT l.parent_user_id FROM parent_student_links l
        JOIN student_attendance_notification_permissions permission ON permission.parent_student_link_id=l.id AND permission.allowed=true
        LEFT JOIN parent_attendance_preferences p ON p.parent_user_id=l.parent_user_id AND p.student_user_id=l.student_user_id
        WHERE l.student_user_id=$1 AND l.status='active' AND COALESCE(p.notify_curfew,true)`, [student.user_id]);
      if (inserted.rowCount) {
        created += 1;
        for (const parent of parents.rows) {
          const result = await client.query(`INSERT INTO notifications(user_id,type,title,body,data)
          SELECT $1,'attendance_exception','Student absent at curfew',
          'Your linked student is recorded outside the hostel after curfew.',$2::jsonb
          WHERE COALESCE((SELECT attendance_updates FROM notification_preferences WHERE user_id=$1),true)`,
          [parent.parent_user_id, JSON.stringify({attendance_exception_id:exceptionId,student_id:student.user_id,hostel_id:setting.hostel_id})]);
          notified += result.rowCount;
        }
        await client.query(`INSERT INTO notifications(user_id,type,title,body,data)
          SELECT wa.warden_user_id,'attendance_exception','Student absent at curfew',
            'A hostel resident is recorded outside after curfew.', $2::jsonb
          FROM warden_assignments wa WHERE wa.hostel_id=$1
            AND COALESCE((SELECT attendance_updates FROM notification_preferences WHERE user_id=wa.warden_user_id),true)`,
        [setting.hostel_id, JSON.stringify({attendance_exception_id:exceptionId,student_id:student.user_id,hostel_id:setting.hostel_id})]);
      }
      if (now.getTime() < new Date(setting.curfew_at).getTime() + Number(setting.escalation_minutes) * 60000) continue;
      const escalated = await client.query("INSERT INTO attendance_curfew_escalations(exception_id) VALUES($1) ON CONFLICT DO NOTHING RETURNING exception_id", [exceptionId]);
      if (!escalated.rowCount) continue;
      const data = JSON.stringify({attendance_exception_id:exceptionId,student_id:student.user_id,hostel_id:setting.hostel_id});
      for (const parent of parents.rows) {
        const result = await client.query(`INSERT INTO notifications(user_id,type,title,body,data)
          SELECT $1,'attendance_escalation','Student still outside after escalation threshold',
          'Your linked student is still recorded outside well after curfew. Please check on them.',$2::jsonb
          WHERE COALESCE((SELECT attendance_updates FROM notification_preferences WHERE user_id=$1),true)`, [parent.parent_user_id, data]);
        notified += result.rowCount;
      }
      await client.query(`INSERT INTO notifications(user_id,type,title,body,data)
        SELECT wa.warden_user_id,'attendance_escalation','Resident still outside after escalation threshold',
          'A resident is still recorded outside well after curfew.', $2::jsonb
        FROM warden_assignments wa WHERE wa.hostel_id=$1
          AND COALESCE((SELECT attendance_updates FROM notification_preferences WHERE user_id=wa.warden_user_id),true)`, [setting.hostel_id, data]);
    }
  }
  return { delivered:true, detail:`${created} curfew exception(s) created; ${notified} parent notification(s) queued`, created, notified };
}

async function processOneJob(pool, workerId) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const claimed = await client.query(`WITH candidate AS (SELECT id FROM operational_jobs WHERE status='queued' AND run_at<=NOW() ORDER BY run_at,id FOR UPDATE SKIP LOCKED LIMIT 1)
      UPDATE operational_jobs j SET status='processing',locked_at=NOW(),locked_by=$1,updated_at=NOW() FROM candidate WHERE j.id=candidate.id RETURNING j.*`, [workerId]);
    if (!claimed.rowCount) { await client.query("COMMIT"); return null; }
    const job = claimed.rows[0]; let outcome;
    if (job.job_type === "emergency_delivery") outcome = await processDelivery(client, job, "emergency_delivery_attempts", "id");
    else if (job.job_type === "fee_reminder_delivery") outcome = await processDelivery(client, job, "fee_reminder_deliveries", "id");
    else if (job.job_type === "condition_deadline_check") outcome = await processConditionDeadlineCheck(client);
    else if (job.job_type === "invoice_overdue_check") outcome = await processInvoiceOverdueCheck(client);
    else if (job.job_type === "fee_invoice_generation") outcome = await processFeeInvoiceGeneration(client);
    else if (job.job_type === "fee_reminder_check") outcome = await processFeeReminderCheck(client);
    else if (job.job_type === "complaint_sla_check") outcome = await processComplaintSlaCheck(client, workerId);
    else if (job.job_type === "complaint_volume_check") outcome = await processComplaintVolumeCheck(client, workerId);
    else if (job.job_type === "location_retention_check") outcome = await processLocationRetentionCheck(client);
    else if (job.job_type === "booking_expiry_check") outcome = await processBookingExpiryCheck(client);
    else if (job.job_type === "complaint_auto_close") outcome = await processComplaintAutoClose(client);
    else if (job.job_type === "attendance_curfew_check") outcome = await processAttendanceCurfewCheck(client);
    else if (job.job_type === "availability_alert_check") outcome = await processAvailabilityAlertCheck(client);
    else outcome = { delivered: false, error: "Unsupported job type" };
    const attemptNumber = Number(job.attempt_count) + 1;
    if (outcome.delivered) {
      await client.query("UPDATE operational_jobs SET status='succeeded',attempt_count=$1,completed_at=NOW(),locked_at=NULL,locked_by=NULL,last_error=NULL,updated_at=NOW() WHERE id=$2", [attemptNumber, job.id]);
      await client.query("INSERT INTO operational_job_attempts (job_id,attempt_number,outcome,details) VALUES ($1,$2,'succeeded',$3::jsonb)", [job.id, attemptNumber, JSON.stringify(outcome)]);
    } else {
      const dead = attemptNumber >= Number(job.max_attempts);
      await client.query("UPDATE operational_jobs SET status=$1,attempt_count=$2,run_at=$3,locked_at=NULL,locked_by=NULL,last_error=$4,updated_at=NOW() WHERE id=$5", [dead ? "dead" : "queued", attemptNumber, dead ? new Date() : retryAt(attemptNumber), outcome.error, job.id]);
      await client.query("INSERT INTO operational_job_attempts (job_id,attempt_number,outcome,details) VALUES ($1,$2,$3,$4::jsonb)", [job.id, attemptNumber, dead ? "dead" : "failed", JSON.stringify(outcome)]);
    }
    await client.query("COMMIT"); return { id: job.id, type: job.job_type, ...outcome };
  } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
}

async function runDueJobs(pool, options = {}) {
  const limit = Math.max(1, Math.min(Number(options.limit || 25), 200));
  const workerId = options.workerId || `worker-${crypto.randomUUID()}`; const results = [];
  for (let index = 0; index < limit; index += 1) { const result = await processOneJob(pool, workerId); if (!result) break; results.push(result); }
  return results;
}

async function enqueueRecurringJobs(pool) {
  await enqueueOperationalJob(pool, "condition_deadline_check", `condition-deadline:${new Date().toISOString().slice(0, 13)}`, {});
  await enqueueOperationalJob(pool, "invoice_overdue_check", `invoice-overdue:${new Date().toISOString().slice(0, 13)}`, {});
  await enqueueOperationalJob(pool, "fee_invoice_generation", `fee-invoice-generation:${new Date().toISOString().slice(0, 10)}`, {});
  await enqueueOperationalJob(pool, "fee_reminder_check", `fee-reminder-check:${new Date().toISOString().slice(0, 10)}`, {});
  await enqueueOperationalJob(pool, "complaint_sla_check", `complaint-sla:${new Date().toISOString().slice(0, 13)}`, {});
  await enqueueOperationalJob(pool, "complaint_volume_check", `complaint-volume:${new Date().toISOString().slice(0, 13)}`, {});
  await enqueueOperationalJob(pool, "location_retention_check", `location-retention:${new Date().toISOString().slice(0, 10)}`, {});
  await enqueueOperationalJob(pool, "booking_expiry_check", `booking-expiry:${new Date().toISOString().slice(0, 13)}`, {});
  await enqueueOperationalJob(pool, "complaint_auto_close", `complaint-auto-close:${new Date().toISOString().slice(0, 13)}`, {});
  await enqueueOperationalJob(pool, "attendance_curfew_check", `attendance-curfew:${new Date().toISOString().slice(0, 16)}`, {});
  await enqueueOperationalJob(pool, "availability_alert_check", `availability-alert:${new Date().toISOString().slice(0, 13)}`, {});
}

module.exports = { runDueJobs, enqueueRecurringJobs };
