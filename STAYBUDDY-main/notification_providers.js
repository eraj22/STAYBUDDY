const nodemailer = require("nodemailer");

function deliveryMode() {
  return String(process.env.NOTIFICATION_DELIVERY_MODE || "fake").trim().toLowerCase();
}

function configured(value) {
  return typeof value === "string" && value.trim().length > 0;
}

async function sendSmtpEmail({ recipient, subject, body }) {
  if (!recipient.email) return { delivered: false, provider: "smtp", error: "Recipient has no email address" };
  if (!configured(process.env.EMAIL_USER) || !configured(process.env.EMAIL_PASS)) return { delivered: false, provider: "smtp", error: "SMTP credentials are not configured" };
  const port = Number(process.env.SMTP_PORT || 587);
  const transport = configured(process.env.SMTP_HOST)
    ? nodemailer.createTransport({ host: process.env.SMTP_HOST, port, secure: String(process.env.SMTP_SECURE || '').toLowerCase() === 'true' || port === 465, auth: { user: process.env.EMAIL_USER, pass: process.env.EMAIL_PASS } })
    : nodemailer.createTransport({ service: process.env.EMAIL_SERVICE || "gmail", auth: { user: process.env.EMAIL_USER, pass: process.env.EMAIL_PASS } });
  const info = await transport.sendMail({ from: process.env.EMAIL_FROM || process.env.EMAIL_USER, to: recipient.email, subject, text: body });
  return { delivered: true, provider: "smtp", providerReference: info.messageId || null };
}

async function sendTwilioSms({ recipient, body }) {
  if (!recipient.phone) return { delivered: false, provider: "twilio", error: "Recipient has no phone number" };
  const accountSid = process.env.TWILIO_ACCOUNT_SID, authToken = process.env.TWILIO_AUTH_TOKEN, from = process.env.TWILIO_FROM_NUMBER;
  if (![accountSid, authToken, from].every(configured)) return { delivered: false, provider: "twilio", error: "Twilio SMS credentials are not configured" };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Number(process.env.NOTIFICATION_PROVIDER_TIMEOUT_MS || 15000));
  try {
    const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${encodeURIComponent(accountSid)}/Messages.json`, {
      method: "POST", signal: controller.signal,
      headers: { Authorization: `Basic ${Buffer.from(`${accountSid}:${authToken}`).toString("base64")}`, "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ To: recipient.phone, From: from, Body: body }).toString(),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) return { delivered: false, provider: "twilio", error: payload.message || `Twilio returned HTTP ${response.status}` };
    return { delivered: true, provider: "twilio", providerReference: payload.sid || null };
  } finally { clearTimeout(timer); }
}

// A provider-neutral push adapter. Point this at a trusted service that
// accepts a device token and notification payload (for example an FCM/Expo
// relay). Keeping the provider URL in environment configuration avoids storing
// notification credentials or device tokens in source code.
async function sendPushWebhook({ recipient, subject, body }) {
  const url = process.env.PUSH_WEBHOOK_URL;
  if (!configured(url)) return { delivered: false, provider: "push_webhook", error: "PUSH_WEBHOOK_URL is not configured" };
  if (!recipient.push_token) return { delivered: false, provider: "push_webhook", error: "Recipient has no registered push token" };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Number(process.env.NOTIFICATION_PROVIDER_TIMEOUT_MS || 15000));
  try {
    const response = await fetch(url, {
      method: "POST", signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(configured(process.env.PUSH_WEBHOOK_BEARER_TOKEN) ? { Authorization: `Bearer ${process.env.PUSH_WEBHOOK_BEARER_TOKEN}` } : {}) },
      body: JSON.stringify({ token: recipient.push_token, title: subject, body }),
    });
    if (!response.ok) return { delivered: false, provider: "push_webhook", error: `Push provider returned HTTP ${response.status}` };
    const payload = await response.json().catch(() => ({}));
    return { delivered: true, provider: "push_webhook", providerReference: payload.id || payload.message_id || null };
  } finally { clearTimeout(timer); }
}

// fake (default) is for local development. smtp, twilio, and auto are
// production modes; auto routes email to SMTP and SMS to Twilio.
async function sendExternalDelivery({ channel, recipient, subject, body }) {
  try {
    const mode = deliveryMode();
    if (mode === "fake") return { delivered: true, provider: "fake", providerReference: `fake-${channel}-${Date.now()}` };
    if (channel === "email") {
      if (!["smtp", "auto"].includes(mode)) return { delivered: false, provider: mode, error: "Email delivery is disabled by NOTIFICATION_DELIVERY_MODE" };
      return await sendSmtpEmail({ recipient, subject, body });
    }
    if (channel === "sms") {
      if (!["twilio", "auto"].includes(mode)) return { delivered: false, provider: mode, error: "SMS delivery is disabled by NOTIFICATION_DELIVERY_MODE" };
      return await sendTwilioSms({ recipient, body });
    }
    if (channel === "push") {
      if (!['push', 'auto'].includes(mode)) return { delivered: false, provider: mode, error: "Push delivery is disabled by NOTIFICATION_DELIVERY_MODE" };
      return await sendPushWebhook({ recipient, subject, body });
    }
    return { delivered: false, provider: mode, error: `${channel} delivery is not supported by the configured provider` };
  } catch (error) {
    return { delivered: false, provider: deliveryMode(), error: error.name === "AbortError" ? "Notification provider timed out" : (error.message || "Provider delivery failed") };
  }
}

module.exports = { sendExternalDelivery, deliveryMode };
