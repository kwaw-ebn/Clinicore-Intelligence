import FIREBASE_CONFIG from './firebase-config.js';

if (!window.firebase) throw new Error('Firebase SDK missing');
const app = firebase.apps.length ? firebase.app() : firebase.initializeApp(FIREBASE_CONFIG);
const auth = app.auth();
const db = app.firestore();
let featureChart;
let currentPredictionId = null;
let currentRequestId = null;

const $ = id => document.getElementById(id);
const escapeHtml = value => String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

async function api(path, body) {
  const response = await fetch(path, {method: body ? 'POST' : 'GET', headers: body ? {'Content-Type':'application/json'} : {}, body: body ? JSON.stringify(body) : undefined});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function payload() {
  return {Age:Number($('age').value), Gender:$('gender').value, Fever:$('fever').checked?'Yes':'No', Cough:$('cough').checked?'Yes':'No', Fatigue:$('fatigue').checked?'Yes':'No', DifficultyBreathing:$('dbreath').checked?'Yes':'No', BloodPressure:$('bp_cat').value, Cholesterol:$('chol').value};
}

function showResult(disease, outcome) {
  const rows = disease.top3.map(item => `<li><strong>${escapeHtml(item.condition)}</strong>: ${(item.confidence*100).toFixed(1)}%</li>`).join('');
  $('predictionResult').innerHTML = `<h3>Model output</h3><ul>${rows}</ul><p><strong>${escapeHtml(outcome.risk)}</strong>: ${(outcome.probability*100).toFixed(1)}%</p><p class="warning">Prototype only. Confidence values are model scores, not diagnostic probabilities. Verify clinically.</p><small>Request: ${escapeHtml(disease.request_id || 'unavailable')}</small>`;
  $('predictionResult').hidden = false;
  $('feedbackForm').hidden = false;
  currentRequestId = disease.request_id || null;
}

function requireConsent() {
  if (!$('pilotConsent').checked) throw new Error('Accept the pilot agreement before testing.');
}

$('predictForm').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.submitter; button.disabled = true;
  try {
    requireConsent();
    const input = payload();
    if (!Number.isFinite(input.Age) || input.Age < 0 || input.Age > 120) throw new Error('Enter an age between 0 and 120.');
    const [disease, outcome] = await Promise.all([api('/predict-disease', input), api('/predict-outcome', input)]);
    showResult(disease, outcome);
    const user = auth.currentUser;
    const record = await db.collection('diagnosis').add({patient_name:$('pname').value.trim() || 'Synthetic test record', features:input, prediction:{diseaseRes:disease,outcomeRes:outcome}, requestId:currentRequestId, modelVersion:disease.model_version || null, createdBy:user.uid, createdAt:firebase.firestore.FieldValue.serverTimestamp()});
    currentPredictionId = record.id;
    await refreshRecords();
  } catch (error) { $('predictionResult').textContent = error.message; $('predictionResult').hidden = false; }
  finally { button.disabled = !$('pilotConsent').checked; }
});

$('runSymptomCheck').addEventListener('click', async () => {
  try { requireConsent(); const [disease, outcome] = await Promise.all([api('/predict-disease', payload()), api('/predict-outcome', payload())]); currentPredictionId = null; showResult(disease, outcome); }
  catch (error) { $('predictionResult').textContent = error.message; $('predictionResult').hidden = false; }
});

$('pilotConsent').addEventListener('change', event => {
  document.querySelectorAll('.pilot-action').forEach(button => { button.disabled = !event.target.checked; });
});

$('savePatientBtn').addEventListener('click', async () => {
  requireConsent();
  const name=$('pf_name').value.trim(), age=Number($('pf_age').value);
  if (!name || !Number.isFinite(age) || age<0 || age>120) return alert('Enter a name and valid age.');
  const user=auth.currentUser;
  await db.collection('patients').add({name,age,gender:$('pf_gender').value,phone:$('pf_phone').value.trim(),blood_pressure:$('pf_bp').value,createdBy:user.uid,createdAt:firebase.firestore.FieldValue.serverTimestamp()});
  alert('Prototype patient profile saved. Do not enter identifiable real patient data during public testing.');
});

$('feedbackForm').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    requireConsent();
    const user = auth.currentUser;
    await db.collection('pilot_feedback').add({
      predictionId: currentPredictionId,
      requestId: currentRequestId,
      rating: $('feedbackRating').value,
      comment: $('feedbackComment').value.trim(),
      createdBy: user.uid,
      createdAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    event.target.reset();
    event.target.hidden = true;
    $('feedbackMessage').textContent = 'Feedback saved. Thank you.';
  } catch (error) {
    $('feedbackMessage').textContent = error.message;
  }
});

async function refreshRecords() {
  const snap=await db.collection('diagnosis').orderBy('createdAt','desc').limit(30).get();
  $('totalRecords').textContent=snap.size;
  const labels=new Set();
  $('recordsList').innerHTML=snap.docs.map(doc=>{const r=doc.data(), top=r.prediction?.diseaseRes?.top3?.[0]?.condition || 'Unavailable'; labels.add(top); return `<div class="record"><strong>${escapeHtml(r.patient_name||'Prototype record')}</strong> · ${escapeHtml(top)}</div>`}).join('') || '<p>No test records yet.</p>';
  $('uniqueDx').textContent=labels.size;
}

async function loadFeatures(){
  try { const data=await api('/feature-importance'); const ctx=$('featureImportanceChart'); featureChart?.destroy(); featureChart=new Chart(ctx,{type:'bar',data:{labels:data.map(x=>x.feature),datasets:[{label:'Relative importance',data:data.map(x=>x.importance)}]},options:{responsive:true}}); }
  catch(error){ console.warn(error); }
}

async function loadStatus() {
  try {
    const data = await api('/meta');
    $('serviceStatus').textContent = data.models_loaded ? 'Service ready' : 'Models unavailable';
    $('serviceStatus').classList.toggle('ok', data.models_loaded);
    $('modelVersion').textContent = data.model_version || 'unknown';
  } catch (error) {
    $('serviceStatus').textContent = 'Service unavailable';
  }
}

async function loadAdminFeedback() {
  const snapshot = await db.collection('pilot_feedback').orderBy('createdAt', 'desc').limit(100).get();
  const rows = snapshot.docs.map(doc => doc.data());
  $('feedbackTotal').textContent = rows.length;
  $('unsafeTotal').textContent = rows.filter(row => row.rating === 'unsafe_or_misleading').length;
  $('feedbackList').innerHTML = rows.slice(0, 20).map(row =>
    `<div class="record"><strong>${escapeHtml(row.rating)}</strong> · ${escapeHtml(row.comment || 'No comment')}</div>`
  ).join('') || '<p>No feedback yet.</p>';
}

function addChat(role,text){const line=document.createElement('div');line.className='chat-line';line.textContent=`${role}: ${text}`;$('chat-box').appendChild(line);}
$('sendChat').addEventListener('click', async()=>{try{requireConsent();const message=$('chatInput').value.trim();if(!message)return;addChat('Clinician',message);$('chatInput').value='';const data=await api('/chat',{message});addChat('Assistant',data.reply);}catch(error){addChat('System',error.message);}});
$('generateNote').addEventListener('click',async()=>{try{requireConsent();const data=await api('/generate-note',{chat:$('chat-box').innerText});$('noteOutput').textContent=data.note;$('noteOutput').hidden=false;}catch(error){alert(error.message);}});
$('logoutBtn').addEventListener('click',()=>auth.signOut());

auth.onAuthStateChanged(async user=>{if(!user){location.href='login.html';return;}$('userName').textContent=user.displayName||user.email;await Promise.all([refreshRecords(),loadFeatures(),loadStatus()]);const roleDoc=await db.collection('users').doc(user.uid).get();const role=roleDoc.exists?roleDoc.data().role:'unassigned';$('roleLabel').textContent=role;if(role==='admin'){$('adminPanel').hidden=false;await loadAdminFeedback();}});
