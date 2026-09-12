import FIREBASE_CONFIG from './firebase-config.js';

if (!window.firebase) throw new Error('Firebase SDK missing');
const app = firebase.apps.length ? firebase.app() : firebase.initializeApp(FIREBASE_CONFIG);
const auth = app.auth();
const db = app.firestore();
let featureChart;
let currentPredictionId = null;
let currentRequestId = null;
let savedRecords = [];

const $ = id => document.getElementById(id);
const escapeHtml = value => String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

async function api(path, body) {
  const response = await fetch(path, {method: body ? 'POST' : 'GET', headers: body ? {'Content-Type':'application/json'} : {}, body: body ? JSON.stringify(body) : undefined});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function payload() {
  const symptomIds=['fever','chills','headache','cough','fatigue','dbreath','sore_throat','runny_nose','nausea','vomiting','diarrhea','abdominal_pain','painful_urination','urinary_frequency','flank_pain','rash','itching','confusion'];
  const symptoms=Object.fromEntries(symptomIds.map(id=>[id,$(id).checked]));
  return {Age:Number($('age').value), Gender:$('gender').value, Fever:symptoms.fever?'Yes':'No', Cough:symptoms.cough?'Yes':'No', Fatigue:symptoms.fatigue?'Yes':'No', DifficultyBreathing:symptoms.dbreath?'Yes':'No', BloodPressure:$('bp_cat').value, Cholesterol:$('chol').value, symptoms, clinicalContext:clinicalContext()};
}

const optionalNumber = id => {
  const raw = $(id).value.trim();
  return raw === '' ? null : Number(raw);
};

function calculateBmi() {
  const weight = optionalNumber('weight_kg');
  const height = optionalNumber('height_cm');
  const bmi = weight && height ? weight / ((height / 100) ** 2) : null;
  $('bmi_output').textContent = Number.isFinite(bmi) ? bmi.toFixed(1) : 'Not calculated';
  return Number.isFinite(bmi) ? Number(bmi.toFixed(1)) : null;
}

function clinicalContext() {
  const glucoseType = $('glucose_type').value;
  return {
    vitalSigns: {temperature_c:optionalNumber('temperature'), pulse_bpm:optionalNumber('pulse'), respiratory_rate_bpm:optionalNumber('respiratory_rate'), spo2_percent:optionalNumber('spo2'), systolic_bp_mmhg:optionalNumber('systolic_bp'), diastolic_bp_mmhg:optionalNumber('diastolic_bp')},
    anthropometry: {weight_kg:optionalNumber('weight_kg'), height_cm:optionalNumber('height_cm'), bmi_kg_m2:calculateBmi()},
    laboratory: {malaria_rdt:$('malaria_rdt').value, hemoglobin_g_dl:optionalNumber('hemoglobin'), wbc_10e9_l:optionalNumber('wbc'), urine_leukocyte_esterase:$('urine_le').value, urine_nitrite:$('urine_nitrite').value},
    glucose: {test_type:glucoseType, result_mmol_l:glucoseType === 'not_done' ? null : optionalNumber('glucose_value')},
    chronicDiseaseHistory: {hypertension:$('hypertension_history').checked, diabetes:$('diabetes_history').checked, other:$('other_chronic_history').checked, other_details:$('other_chronic_history').checked ? $('other_chronic_details').value.trim() : ''}
  };
}

function validateClinicalContext(context) {
  if ((context.anthropometry.weight_kg === null) !== (context.anthropometry.height_cm === null)) throw new Error('Enter both weight and height to calculate BMI, or leave both blank.');
  if (context.glucose.test_type !== 'not_done' && context.glucose.result_mmol_l === null) throw new Error('Enter the measured glucose result for the selected FBS or RBS test.');
}

['weight_kg','height_cm'].forEach(id => $(id).addEventListener('input', calculateBmi));
$('glucose_type').addEventListener('change', event => { $('glucose_value').disabled = event.target.value === 'not_done'; if (event.target.value === 'not_done') $('glucose_value').value = ''; });
$('other_chronic_history').addEventListener('change', event => { $('other_chronic_details').disabled = !event.target.checked; if (!event.target.checked) $('other_chronic_details').value = ''; });

function showResult(disease, outcome) {
  const rows = disease.top3.map(item => `<li><strong>${escapeHtml(item.condition)}</strong>: ${(item.confidence*100).toFixed(1)}%</li>`).join('');
  $('predictionResult').innerHTML = `<h3>Synthetic model output</h3><p><strong>Stage:</strong> ${escapeHtml(disease.model_stage === 'post_lab' ? 'Post-lab support' : 'Pre-lab pattern')}</p><ul>${rows}</ul><p><strong>${escapeHtml(outcome.risk)}</strong>: ${(outcome.probability*100).toFixed(1)}%</p><p class="warning">Prototype only. This model was trained entirely on synthetic scenarios. Scores are not diagnostic probabilities or evidence of clinical accuracy. Verify clinically.</p><small>Request: ${escapeHtml(disease.request_id || 'unavailable')}</small>`;
  $('predictionResult').hidden = false;
  $('feedbackForm').hidden = false;
  currentRequestId = disease.request_id || null;
}

function clearOutput() {
  $('predictionResult').hidden = true;
  $('predictionResult').innerHTML = '';
  $('feedbackForm').hidden = true;
  $('feedbackForm').reset();
  $('feedbackMessage').textContent = '';
  currentPredictionId = null;
  currentRequestId = null;
}

function showSavedRecord(record) {
  const disease = record.prediction?.diseaseRes;
  const outcome = record.prediction?.outcomeRes;
  if (!disease?.top3 || !outcome) throw new Error('This saved record has no readable model output.');
  showResult(disease, outcome);
  currentPredictionId = record.id;
  currentRequestId = record.requestId || disease.request_id || null;
  $('predictionResult').scrollIntoView({behavior:'smooth', block:'center'});
}

$('clearOutputBtn').addEventListener('click', clearOutput);
$('findRecordBtn').addEventListener('click', () => {
  const query = $('recordSearch').value.trim().toLowerCase();
  const message = $('recordSearchMessage');
  if (!query) { message.textContent = 'Enter a recorded label first.'; return; }
  const exact = savedRecords.find(record => (record.patient_name || '').trim().toLowerCase() === query);
  const partial = savedRecords.find(record => (record.patient_name || '').toLowerCase().includes(query));
  const record = exact || partial;
  if (!record) { message.textContent = 'No saved output found with that label.'; return; }
  try { showSavedRecord(record); message.textContent = `Showing ${record.patient_name}.`; }
  catch (error) { message.textContent = error.message; }
});
$('recordSearch').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); $('findRecordBtn').click(); } });

function requireConsent() {
  if (!$('pilotConsent').checked) throw new Error('Accept the pilot agreement before testing.');
}

$('predictForm').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.submitter; button.disabled = true;
  try {
    requireConsent();
    const input = payload();
    if (!Number.isFinite(input.Age) || input.Age < 18 || input.Age > 120) throw new Error('This adult MVP requires an age between 18 and 120.');
    validateClinicalContext(input.clinicalContext);
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
  try { requireConsent(); const input=payload(); validateClinicalContext(input.clinicalContext); const [disease, outcome] = await Promise.all([api('/predict-disease', input), api('/predict-outcome', input)]); currentPredictionId = null; showResult(disease, outcome); }
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
  const user = auth.currentUser;
  if (!user) return;
  const snap = await db.collection('diagnosis')
    .where('createdBy', '==', user.uid)
    .limit(50)
    .get();
  const docs = [...snap.docs].sort((a, b) => {
    const aTime = a.data().createdAt?.toMillis?.() || 0;
    const bTime = b.data().createdAt?.toMillis?.() || 0;
    return bTime - aTime;
  }).slice(0, 30);
  savedRecords = docs.map(doc => ({id:doc.id, ...doc.data()}));
  $('totalRecords').textContent = docs.length;
  const labels = new Set();
  $('recordsList').innerHTML = savedRecords.map(record => {
    const top = record.prediction?.diseaseRes?.top3?.[0]?.condition || 'Unavailable';
    labels.add(top);
    return `<button type="button" class="record record-button" data-record-id="${escapeHtml(record.id)}"><strong>${escapeHtml(record.patient_name || 'Prototype record')}</strong><span>${escapeHtml(top)}</span></button>`;
  }).join('') || '<p>No test records yet.</p>';
  $('recordsList').querySelectorAll('[data-record-id]').forEach(button => button.addEventListener('click', () => {
    const record = savedRecords.find(item => item.id === button.dataset.recordId);
    if (record) showSavedRecord(record);
  }));
  $('uniqueDx').textContent = labels.size;
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
$('sendChat').addEventListener('click', async()=>{try{requireConsent();const message=$('chatInput').value.trim();if(!message)return;addChat('Clinician',message);$('chatInput').value='';const data=await api('/chat',{message});addChat(data.mode==='demo'?'Demo assistant':'AI assistant',data.reply);}catch(error){addChat('System',error.message);}});
$('generateNote').addEventListener('click',async()=>{try{requireConsent();const data=await api('/generate-note',{chat:$('chat-box').innerText});$('noteOutput').textContent=data.note;$('noteOutput').hidden=false;}catch(error){alert(error.message);}});
$('logoutBtn').addEventListener('click',()=>auth.signOut());

auth.onAuthStateChanged(async user=>{if(!user){location.href='login.html';return;}$('userName').textContent=user.displayName||user.email;await Promise.all([refreshRecords(),loadFeatures(),loadStatus()]);const roleDoc=await db.collection('users').doc(user.uid).get();const role=roleDoc.exists?roleDoc.data().role:'unassigned';$('roleLabel').textContent=role;if(role==='admin'){$('adminPanel').hidden=false;await loadAdminFeedback();}});
