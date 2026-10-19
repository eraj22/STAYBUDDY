const text=(v,max=4000)=>typeof v==='string'?v.trim().slice(0,max):'';
function registerChatSupportRoutes({app,pool,requireAuth,requireRole,isValidInt}) {
  const admin=[requireAuth,requireRole('admin')];
  app.get('/api/admin/chat/escalations/:id/messages',...admin,async(req,res)=>{
    if(!isValidInt(Number(req.params.id))) return res.status(400).json({error:'Invalid escalation id'});
    const e=await pool.query('SELECT session_id FROM chatbot_support_escalations WHERE id=$1',[req.params.id]);
    if(!e.rowCount) return res.status(404).json({error:'Escalation not found'});
    const messages=await pool.query('SELECT * FROM chat_messages WHERE session_id=$1 ORDER BY created_at,id',[e.rows[0].session_id]);
    res.json({messages:messages.rows});
  });
  app.post('/api/admin/chat/escalations/:id/messages',...admin,async(req,res)=>{
    const content=text(req.body?.message);
    if(!content||!isValidInt(Number(req.params.id))) return res.status(400).json({error:'Valid escalation id and message required'});
    const client=await pool.connect();
    try {
      await client.query('BEGIN');
      const e=await client.query('SELECT * FROM chatbot_support_escalations WHERE id=$1 FOR UPDATE',[req.params.id]);
      if(!e.rowCount) {await client.query('ROLLBACK');return res.status(404).json({error:'Escalation not found'});}
      if(!['open','in_progress'].includes(e.rows[0].status)) {await client.query('ROLLBACK');return res.status(409).json({error:'Support request is closed'});}
      const session=e.rows[0].session_id;
      await client.query('SELECT pg_advisory_xact_lock(hashtextextended($1,0))',[session]);
      const message=await client.query("INSERT INTO chat_messages(session_id,role,content,metadata) VALUES($1,'assistant',$2,$3::jsonb) RETURNING *",[session,content,JSON.stringify({provider:'human_support',agent_user_id:req.user.id,escalation_id:e.rows[0].id})]);
      await client.query("UPDATE chatbot_support_escalations SET status='in_progress',assigned_to_user_id=$2,updated_at=NOW() WHERE id=$1",[req.params.id,req.user.id]);
      await client.query('UPDATE chat_sessions SET updated_at=NOW() WHERE id=$1',[session]);
      await client.query('COMMIT');res.status(201).json({message:message.rows[0]});
    } catch(error) {await client.query('ROLLBACK');throw error;} finally {client.release();}
  });
  app.post('/api/chat/messages/:id/report',requireAuth,requireRole('student','parent','warden'),async(req,res)=>{
    const description=text(req.body?.description,2000);
    if(!description||!isValidInt(Number(req.params.id))) return res.status(400).json({error:'A message id and description are required'});
    const owned=await pool.query("SELECT m.session_id FROM chat_messages m JOIN chat_sessions s ON s.id=m.session_id WHERE m.id=$1 AND s.user_id=$2 AND m.role='assistant'",[req.params.id,req.user.id]);
    if(!owned.rowCount) return res.status(404).json({error:'Message not found'});
    const report=await pool.query('INSERT INTO chatbot_discrepancies(session_id,user_id,message_id,description) VALUES($1,$2,$3,$4) RETURNING *',[owned.rows[0].session_id,req.user.id,req.params.id,description]);
    res.status(201).json({report:report.rows[0]});
  });
  app.get('/api/admin/chat/reports',...admin,async(_req,res)=>{
    res.json({reports:(await pool.query('SELECT * FROM chatbot_discrepancies ORDER BY created_at DESC LIMIT 200')).rows});
  });
  app.patch('/api/admin/chat/reports/:id',...admin,async(req,res)=>{
    const note=text(req.body?.resolution_note);
    if(!note||!isValidInt(Number(req.params.id))) return res.status(400).json({error:'Valid id and resolution_note required'});
    const r=await pool.query("UPDATE chatbot_discrepancies SET status='resolved',resolution_note=$2,reviewed_by_user_id=$3,resolved_at=NOW() WHERE id=$1 RETURNING *",[req.params.id,note,req.user.id]);
    if(!r.rowCount)return res.status(404).json({error:'Report not found'});res.json({report:r.rows[0]});
  });
  app.patch('/api/admin/chat/knowledge/:id',...admin,async(req,res)=>{
    const b=req.body||{},id=Number(req.params.id);
    const chunks=b.chunks;
    if(!isValidInt(id)||(b.active!=null&&typeof b.active!=='boolean') ||
       (chunks!==undefined&&(!Array.isArray(chunks)||!chunks.length||chunks.length>30||chunks.some(c=>typeof c!=='string'||c.trim().length<20||c.length>8000))))
      return res.status(400).json({error:'Valid document fields and 1–30 chunks of 20–8000 characters required'});
    const client=await pool.connect();
    try {
      await client.query('BEGIN');
      const doc=await client.query('SELECT * FROM chatbot_knowledge_documents WHERE id=$1 FOR UPDATE',[id]);
      if(!doc.rowCount){await client.query('ROLLBACK');return res.status(404).json({error:'Knowledge document not found'});}
      const prior=await client.query('SELECT * FROM chatbot_knowledge_chunks WHERE document_id=$1 ORDER BY chunk_index',[id]);
      await client.query('INSERT INTO chatbot_knowledge_revisions(document_id,version,snapshot,actor_user_id) VALUES($1,$2,$3::jsonb,$4)',[id,doc.rows[0].version,JSON.stringify({document:doc.rows[0],chunks:prior.rows}),req.user.id]);
      if(chunks) {
        await client.query('DELETE FROM chatbot_knowledge_chunks WHERE document_id=$1',[id]);
        for(let n=0;n<chunks.length;n++) await client.query('INSERT INTO chatbot_knowledge_chunks(document_id,chunk_index,content) VALUES($1,$2,$3)',[id,n,chunks[n].trim()]);
      }
      const updated=await client.query('UPDATE chatbot_knowledge_documents SET title=COALESCE($2,title),category=COALESCE($3,category),active=COALESCE($4,active),version=version+1,reviewed_at=NOW(),updated_at=NOW() WHERE id=$1 RETURNING *',[id,text(b.title,240)||null,text(b.category,80)||null,b.active??null]);
      await client.query('COMMIT');res.json({document:updated.rows[0]});
    }catch(error){await client.query('ROLLBACK');throw error;}finally{client.release();}
  });
  app.get('/api/admin/chat/analytics',...admin,async(_req,res)=>{
    const counts=await pool.query(`SELECT COUNT(*)::int AS turns,COUNT(*) FILTER(WHERE provider='node_rag_fallback')::int AS fallbacks,
      COUNT(*) FILTER(WHERE escalation_recommended)::int AS escalation_recommendations FROM chatbot_retrieval_events WHERE created_at>NOW()-INTERVAL '30 days'`);
    const feedback=await pool.query('SELECT helpful,COUNT(*)::int AS count FROM chatbot_message_feedback GROUP BY helpful');
    const topics=await pool.query("SELECT detected_intent,COUNT(*)::int AS count FROM chatbot_retrieval_events WHERE created_at>NOW()-INTERVAL '30 days' GROUP BY detected_intent ORDER BY count DESC");
    res.json({last_30_days:counts.rows[0],feedback:feedback.rows,topics:topics.rows});
  });
  app.post('/api/chat/complaints/:id/comments',requireAuth,requireRole('student'),async(req,res)=>{
    const content=text(req.body?.message,2000),id=Number(req.params.id);
    if(!content||!isValidInt(id))return res.status(400).json({error:'Valid complaint id and message required'});
    const comment=await pool.query(`INSERT INTO chatbot_complaint_comments(complaint_id,user_id,content)
      SELECT id,$2,$3 FROM complaints WHERE id=$1 AND user_id=$2 RETURNING *`,[id,req.user.id,content]);
    if(!comment.rowCount)return res.status(404).json({error:'Complaint not found'});
    res.status(201).json({comment:comment.rows[0]});
  });
  app.get('/api/chat/complaints/:id/comments',requireAuth,async(req,res)=>{
    const id=Number(req.params.id);
    if(!isValidInt(id))return res.status(400).json({error:'Invalid complaint id'});
    const access=await pool.query(`SELECT c.id FROM complaints c WHERE c.id=$1 AND
      (c.user_id=$2 OR ($3='warden' AND EXISTS(SELECT 1 FROM warden_assignments WHERE hostel_id=c.hostel_id AND warden_user_id=$2))
      OR ($3='owner' AND EXISTS(SELECT 1 FROM hostel_owners WHERE hostel_id=c.hostel_id AND owner_user_id=$2)))`,[id,req.user.id,req.user.role]);
    if(!access.rowCount)return res.status(404).json({error:'Complaint not found'});
    res.json({comments:(await pool.query('SELECT * FROM chatbot_complaint_comments WHERE complaint_id=$1 ORDER BY created_at,id',[id])).rows});
  });
}
module.exports={registerChatSupportRoutes};
