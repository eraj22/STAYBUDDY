// LLM output is an untrusted query proposal. Only these filters reach fixed SQL.
const INTENTS = new Set(['general_support','general_knowledge','booking','cancellation','fees','complaint_status','attendance','hostel_search','parent_access','human_support','policies','app_help']);
const PERSONAL_INTENTS = new Set(['booking','cancellation','fees','complaint_status','attendance','parent_access']);
const LANGUAGES = new Set(['en','ur','hi','roman_ur']);
const CATEGORIES = {booking:['booking'], cancellation:['booking'], fees:['fees'], complaint_status:['complaints'], attendance:['attendance'], parent_access:['parents'], human_support:['support'], policies:['policies','booking'], app_help:['app','parents']};
const text = (value, max=4000) => typeof value === 'string' ? value.trim().slice(0,max) : '';

function languageFor(message, history=[], requested) {
  if (LANGUAGES.has(requested)) return requested;
  if (/roman\s*urdu/i.test(message)) return 'roman_ur';
  if (/\b(in|speak|reply|answer)\s+(?:in\s+)?english\b/i.test(message)) return 'en';
  if (/\burdu\b/i.test(message) || /[؀-ۿ]/.test(message)) return 'ur';
  if (/\bhindi\b/i.test(message) || /[ऀ-ॿ]/.test(message)) return 'hi';
  if (/\b(kaise|kya|mujhe|mera|meri|chahiye|batao|hain|hai)\b/i.test(message)) return 'roman_ur';
  if (/^(yes|no|ok|okay|thanks|please|and that\??)$/i.test(message.trim())) {
    const previous = history.filter(h=>h.role==='user').at(-1);
    if (previous) return languageFor(previous.content);
  }
  return 'en';
}

function intentFor(message) {
  const v=message.toLowerCase();
  if (/human|support agent|representative|انسان|ایجنٹ/.test(v)) return 'human_support';
  if (/complaint|grievance|شکایت|shikayat/.test(v)) return 'complaint_status';
  if (/attendance|curfew|\bqr\b|حاضری|hazri/.test(v)) return 'attendance';
  if (/cancel|refund|منسوخ/.test(v)) return 'cancellation';
  if (/fee|invoice|receipt|payment|balance|فیس|رسید/.test(v)) return 'fees';
  if (/book|document|reservation|بکنگ|بک|बुक/.test(v)) return 'booking';
  if (/saved search|search history|favorite|favourite|wishlist/.test(v)) return 'app_help';
  if (/find|search|show|compare|cheaper|under\s*\d|near\b|تلاش|chahiye/.test(v)) return 'hostel_search';
  if (/policy|policies|rules|deposit/.test(v)) return 'policies';
  if (/parent|child|consent|linked/.test(v)) return 'parent_access';
  if (/remind|notification|login|password|location|not working/.test(v)) return 'app_help';
  return 'general_support';
}

function safeFilters(value={}) {
  if (!value || typeof value !== 'object') value={};
  const price=value.max_price;
  return {location:text(value.location,120) || null,
    max_price: typeof price === 'number' && Number.isFinite(price) && price>=0 && price<=10000000 ? price : null,
    gender:['boys','girls'].includes(value.gender) ? value.gender : null,
    amenities:Array.isArray(value.amenities) ? [...new Set(value.amenities.map(a=>text(a,40).toLowerCase()).filter(Boolean))].slice(0,6) : [],
    sort:value.sort==='price' ? 'price' : 'rating'};
}

function namedHostelIn(message) {
  const match=message.match(/\b(?:in|at|for|of)\s+((?:[a-z][a-z0-9'-]*\s+){1,5}hostel)\b/i);
  if(!match) return null;
  const name=match[1].replace(/^(?:the|a|an)\s+/i,'').trim();
  const distinctive=name.toLowerCase().split(/\s+/).some(word=>
    !['hostel','girls','boys','female','male','student','students','single','room','shared','private'].includes(word));
  return distinctive ? name : null;
}

function currentSearchFilters(message,proposed) {
  const filters=safeFilters(proposed);
  const hostel=namedHostelIn(message);
  if(hostel) {
    if(!/^(and|what about|also|only)\b/i.test(message.trim())) {
      const current=fallbackPlan(message).filters;
      return {...current,location:hostel};
    }
    filters.location=hostel;
  }
  return filters;
}

function fallbackPlan(message,history=[],previous={},appContext={}) {
  let intent=intentFor(message);
  const followup=/^(and\b|what about\b|only\b|cheaper\b|with\b|without\b|under\b|compare\b|those\b|them\b|yes\b|no\b)/i.test(message);
  if(intent==='general_support' && followup && INTENTS.has(previous.intent)) intent=previous.intent;
  if(intent==='general_support' && appContext.screen) intent=({fees:'fees',booking:'booking',attendance:'attendance',complaints:'complaint_status',search:'hostel_search'})[appContext.screen] || intent;
  const filters=safeFilters(followup && intent==='hostel_search' ? previous.filters : {});
  const money=message.match(/(?:under|below|budget|max(?:imum)?)\s*(?:pkr|rs\.?\s*)?\s*([\d,]+(?:\.\d+)?)\s*(k)?\b/i);
  if(money) filters.max_price=Number(money[1].replaceAll(',',''))*(money[2]?1000:1);
  if(/no budget|any price|no price limit/i.test(message)) filters.max_price=null;
  const place=message.match(/(?:near|in)\s+(.+?)(?=\s+(?:under|below|with|for|budget)\b|[?.!]|$)/i);
  if(place) filters.location=place[1].trim();
  if(/girls|female/i.test(message)) filters.gender='girls'; else if(/boys|male/i.test(message)) filters.gender='boys';
  for(const amenity of ['wifi','parking','mess','ac']) if(new RegExp('\\b'+amenity+'\\b','i').test(message)) filters.amenities=[...new Set([...filters.amenities,amenity])];
  if(/cheap|lowest|price/i.test(message)) filters.sort='price';
  return {intent,language:languageFor(message,history),query:message,
    needs_personal_data:/\b(my|mine|our|child|son|daughter)\b|\b(?:do i have|have i paid|am i overdue|what do i owe|i owe|i have due|i have unpaid)\b|میری|میرا|میرے|meri|mera/i.test(message),
    discrepancy:/outdated|incorrect|doesn.t match|says something different|wrong (information|answer)/i.test(message),
    filters:safeFilters(filters),provider:'rules_fallback'};
}

function safeAppContext(value) {
  if(!value || typeof value!=='object') return {};
  const screens=['booking','fees','attendance','complaints','search','profile','location'];
  return {screen:screens.includes(value.screen)?value.screen:null,
    dwell_seconds:Number.isFinite(value.dwell_seconds)?Math.max(0,Math.min(3600,value.dwell_seconds)):0};
}

async function serviceCall(path,payload,timeout,signal) {
  const base=process.env.AI_CHAT_URL || 'http://127.0.0.1:8000/chat';
  const requestSignal=signal ? AbortSignal.any([signal,AbortSignal.timeout(timeout)]) : AbortSignal.timeout(timeout);
  const response=await fetch(base.replace(/\/$/,'')+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:requestSignal});
  if(!response.ok) throw new Error('service_http_'+response.status);
  return response.json();
}

async function understand(message,history,previous,appContext,requestedLanguage,modelProvider='ollama',signal) {
  const fallback=fallbackPlan(message,history,previous,appContext);
  // Obvious social messages need generation, but no expensive retrieval planning.
  if (/^(hi|hello|hey|salam|assalamualaikum|thanks|thank you)[!.\s]*$/i.test(message.trim())) {
    return {...fallback,intent:'general_support',needs_personal_data:false,language:languageFor(message,history,requestedLanguage),provider:'social_message'};
  }
  const faq = simpleFaq(message, history);
  if (faq) return {...fallback,...faq,needs_personal_data:false,filters:safeFilters(),language:languageFor(message,history,requestedLanguage),provider:'faq_router'};
  // Standalone English how-to questions can go straight to retrieval and the
  // answer LLM. Keep the planner for follow-ups, searches and private records.
  if (!history.length && message.length <= 220 && /^[\x00-\x7F]*$/.test(message)
      && languageFor(message, history, requestedLanguage) === 'en'
      && /^(how|what|why|when|can|does|do|is|are|tell me|explain)\b/i.test(message.trim())
      && /\b(hostel|staybuddy|warden|roommate|curfew|qr attendance|hostel fees|rent|receipt verification|booking documents|saved searches|favorites|favourites)\b/i.test(message)
      && !fallback.needs_personal_data && !fallback.discrepancy
      && !['general_support','hostel_search','human_support'].includes(fallback.intent)
      && !/\b(find|search for|compare|near|under|cheaper|available|recommend)\b/i.test(message)) {
    return {...fallback,provider:'english_howto_router'};
  }
  try {
    const proposed=await serviceCall('/plan',{message,history:history.filter(h=>h.role==='user').slice(-8).map(h=>({role:'user',content:text(h.content,1500)})),previous_plan:previous,app_context:appContext,provider:modelProvider},Number(process.env.AI_CHAT_PLAN_TIMEOUT_MS||190000),signal);
    if(!INTENTS.has(proposed.intent)||!text(proposed.query,600)) throw new Error('invalid_plan');
    return {...fallback,intent:proposed.intent,query:text(proposed.query,600),filters:proposed.intent==='hostel_search' ? currentSearchFilters(message,proposed.filters) : safeFilters(proposed.filters),
      language:LANGUAGES.has(requestedLanguage)?requestedLanguage:LANGUAGES.has(proposed.language)?proposed.language:fallback.language,
      needs_personal_data:PERSONAL_INTENTS.has(proposed.intent) && (fallback.needs_personal_data||proposed.needs_personal_data===true),
      discrepancy:fallback.discrepancy||proposed.discrepancy===true,provider:modelProvider};
  } catch(error) {
    if(signal?.aborted) throw error;
    console.warn('Chat interpretation unavailable:',error.name); // no message/context logging
    return {...fallback,needs_personal_data:PERSONAL_INTENTS.has(fallback.intent)&&fallback.needs_personal_data,language:languageFor(message,history,requestedLanguage)};
  }
}

function simpleFaq(message, history=[]) {
  // Route only standalone procedural questions. Account lookups, ambiguous or
  // mixed-topic messages still use the LLM planner. Answers always use the LLM.
  if(history.length || message.length>180 || /[0-9۰-۹٠-٩]|\bbalance\b|\bstatus\b|بقایا|حیثیت/i.test(message)) return null;
  if(!/\bhow\b|کیسے|\b(?:kaise|kese|kesay)\b|कैसे/i.test(message)) return null;
  const fee=/\bfees?\b|فیس|फीस/i.test(message);
  const booking=/\bbook(?:ing)?\b|بکنگ|بک\b|बुक/i.test(message);
  if(fee && !booking && /\bpay\b|ادا|جمع|\b(?:jama|ada)\b|भुगतान|जमा/i.test(message))
    return {intent:'fees',query:'fees external payment receipt verification'};
  if(booking && !fee && /\bhostel\b|ہاسٹل|होस्टल/i.test(message))
    return {intent:'booking',query:'hostel booking process required documents staff confirmation'};
  return null;
}

const STOP=new Set('a an the to for how do i you we can could should would please about what which is are it this that and or of in my me need want does with have has on at any'.split(' '));
function retrievalTokens(query) {
  return [...new Set(query.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(t=>t.length>2&&!STOP.has(t)))].slice(0,20);
}
function publicHistory(value) {
  return Array.isArray(value)?value.filter(v=>v && typeof v==='object' && ['user','assistant'].includes(v.role) && typeof v.content==='string').slice(-12).map(v=>({role:v.role,content:text(v.content,1500)})):[];
}
module.exports={CATEGORIES,languageFor,intentFor,safeFilters,namedHostelIn,currentSearchFilters,fallbackPlan,safeAppContext,understand,serviceCall,retrievalTokens,publicHistory,simpleFaq};
