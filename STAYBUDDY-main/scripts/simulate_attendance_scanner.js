const crypto=require("node:crypto");

const baseUrl=process.env.STAYBUDDY_API_URL||"http://127.0.0.1:5000";
const key=process.env.ATTENDANCE_INTEGRATION_KEY;
const identifier=process.env.ATTENDANCE_IDENTIFIER;
const direction=process.argv[2]||"in";
const source=process.argv[3]||"rfid";
if(!key||!identifier||!["in","out"].includes(direction)||!["rfid","biometric"].includes(source)){
  console.error("Set ATTENDANCE_INTEGRATION_KEY and ATTENDANCE_IDENTIFIER; use: node scripts/simulate_attendance_scanner.js in|out rfid|biometric");process.exit(1);
}
fetch(`${baseUrl}/api/attendance/events`,{method:"POST",headers:{"Content-Type":"application/json","X-Attendance-Key":key,"Idempotency-Key":crypto.randomUUID()},body:JSON.stringify({external_identifier:identifier,direction,source,occurred_at:new Date().toISOString(),entry_point:"Phase 3 scanner simulator",metadata:{simulated:true}})}).then(async response=>{const body=await response.json();console.log(JSON.stringify(body,null,2));if(!response.ok)process.exitCode=1;}).catch(error=>{console.error(error.message);process.exitCode=1;});
