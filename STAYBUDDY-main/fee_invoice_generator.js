function isBillingMonth(value) {
  return /^\d{4}-\d{2}-01$/.test(String(value || ""));
}

function currentBillingMonth(now = new Date()) {
  return `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, "0")}-01`;
}

function monthEnd(billingMonth) {
  const [year, month] = billingMonth.split("-").map(Number);
  return `${year}-${String(month).padStart(2, "0")}-${new Date(Date.UTC(year, month, 0)).getUTCDate()}`;
}

async function generateFeeInvoices(client, { billingMonth, scheduleId = null }) {
  if (!isBillingMonth(billingMonth)) throw new Error("billingMonth must be YYYY-MM-01");
  const schedules = await client.query("SELECT * FROM hostel_fee_schedules WHERE active=true AND ($1::bigint IS NULL OR id=$1) ORDER BY id", [scheduleId]);
  let created = 0, eligibleAssignments = 0;
  for (const schedule of schedules.rows) {
    const assignments = await client.query("SELECT * FROM student_fee_assignments WHERE schedule_id=$1 AND active=true AND starts_on <= $2::date AND (ends_on IS NULL OR ends_on >= $3::date)", [schedule.id, monthEnd(billingMonth), billingMonth]);
    eligibleAssignments += assignments.rowCount;
    const dueDate = `${billingMonth.slice(0, 8)}${String(schedule.due_day).padStart(2, "0")}`;
    for (const assignment of assignments.rows) {
      const result = await client.query("INSERT INTO fee_invoices (assignment_id,billing_month,due_date,amount,currency) VALUES ($1,$2,$3,$4,$5) ON CONFLICT (assignment_id,billing_month) DO NOTHING RETURNING id", [assignment.id, billingMonth, dueDate, assignment.amount_override || schedule.amount, schedule.currency]);
      created += result.rowCount;
    }
  }
  return { billingMonth, schedules: schedules.rowCount, eligibleAssignments, created };
}

module.exports = { currentBillingMonth, generateFeeInvoices, isBillingMonth };
