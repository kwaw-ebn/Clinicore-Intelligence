// dashboard.js - dashboard logic (feature importance, ROC, confusion matrix, predictions, metrics logging)
import FIREBASE_CONFIG from './firebase-config.js';
if (!window.firebase) throw new Error('Firebase SDK missing');
firebase.initializeApp(FIREBASE_CONFIG);
const db = firebase.firestore();

const API_BASE = '/predict-disease'.startsWith('/') ? '' : ''; // keep empty - fetch uses full endpoints
// We'll call relative endpoints e.g. /predict-disease, /feature-importance etc.

let diagChart = null;
let featChart = null;
let rocChart = null;

// UTIL: simple fetch wrapper
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  return res.json();
}

// LOAD feature importance and render chart
async function loadFeatureImportance(){
  try {
    const res = await fetch('/feature-importance');
    const data = await res.json();
    const labels = data.map(d=>d.feature);
    const values = data.map(d=>d.importance);
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    if (featChart) featChart.destroy();
    featChart = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ label:'Importance', data: values }]},
      options: { responsive:true }
    });
  } catch(err){
    console.error('Feature importance load error', err);
  }
}

// LOAD distribution and records
async function refreshStats(){
  try {
    const snap = await db.collection('diagnosis').orderBy('createdAt','desc').limit(200).get();
    const docs = snap.docs.map(d => d.data());
    document.getElementById('totalRecords').innerText = docs.length;
    const dist = {};
    docs.forEach(r => {
      const lab = (r.prediction && r.prediction.disease_top) || 'Unknown';
      dist[lab] = (dist[lab] || 0) + 1;
    });
    const labels = Object.keys(dist);
    const data = Object.values(dist);
    const ctx = document.getElementById('diagChart').getContext('2d');
    if (diagChart) diagChart.destroy();
    diagChart = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ label:'Count', data }]},
      options: { responsive:true }
    });

    document.getElementById('uniqueDx').innerText = labels.length;
    // recent records list
    const list = document.getElementById('recordsList');
    list.innerHTML = docs.slice(0,30).map(r=>`<div class="record"><strong>${r.patient_name||'Unknown'}</strong> — ${r.prediction ? r.prediction.disease_top : '—'} <br><small>${new Date((r.createdAt && r.createdAt.seconds) ? r.createdAt.seconds*1000 : (r.createdAt || Date.now())).toLocaleString()}</small></div>`).join('');
  } catch(err){
    console.error('refreshStats error', err);
  }
}

// PREDICT form handling
document.getElementById('predictForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const payload = {
    PatientName: document.getElementById('pname').value || '',
    Age: Number(document.getElementById('age').value || 0),
    Gender: document.getElementById('gender').value,
    Fever: document.getElementById('fever').checked ? 'Yes' : 'No',
    Cough: document.getElementById('cough').checked ? 'Yes' : 'No',
    Fatigue: document.getElementById('fatigue').checked ? 'Yes' : 'No',
    DifficultyBreathing: document.getElementById('dbreath').checked ? 'Yes' : 'No',
    temp: Number(document.getElementById('temp').value || 0),
    hr: Number(document.getElementById('hr').value || 0),
    rr: Number(document.getElementById('rr').value || 0),
    spo2: Number(document.getElementById('spo2').value || 100),
    BloodPressure: document.getElementById('bp_cat').value,
    Cholesterol: document.getElementById('chol').value
  };

  // call disease predict
  const diseaseRes = await postJSON('/predict-disease', payload);
  // call outcome predict
  const outcomeRes = await postJSON('/predict-outcome', payload);

  // show immediate result via alert (or better UI)
  const top = diseaseRes.top3 && diseaseRes.top3[0] ? `${diseaseRes.top3[0].disease} (${(diseaseRes.top3[0].confidence*100).toFixed(1)}%)` : 'No result';
  alert(`Top diagnosis: ${top}\nRisk: ${outcomeRes.risk} (${(outcomeRes.probability*100).toFixed(1)}%)`);

  // Save to Firestore
  const authUser = firebase.auth().currentUser;
  await db.collection('diagnosis').add({
    patient_name: payload.PatientName,
    age: payload.Age,
    features: payload,
    prediction: { diseaseRes, outcomeRes },
    createdAt: firebase.firestore.FieldValue.serverTimestamp(),
    createdBy: authUser ? authUser.uid : null
  });

  // Log metrics to server for model monitoring
  try {
    await postJSON('/log-metrics', {
      model: 'disease_model',
      payload,
      prediction: diseaseRes.top3,
      user: authUser ? authUser.uid : null
    });
  } catch(err){
    console.warn('metrics log failed', err);
  }

  await refreshStats();
  await loadFeatureImportance();
});

// ROC & Confusion matrix endpoints
// For ROC/confusion we expect precomputed arrays (y_true, y_prob/y_pred).
// Provide a small utility to fetch metrics from Firestore (last N records) and compute ROC client-side by posting to /roc-data.
async function generateModelMetricsFromRecords(limit=200){
  const snap = await db.collection('diagnosis').orderBy('createdAt','desc').limit(limit).get();
  const docs = snap.docs.map(d => d.data());
  // For simplicity we will attempt to extract binary true labels if available; else skip
  const y_true = [];
  const y_prob = [];
  const y_pred = [];
  for (const r of docs){
    if (!r.features || !r.prediction) continue;
    // if outcome label exists in prediction
    const out = r.prediction && r.prediction.outcomeRes ? r.prediction.outcomeRes : r.prediction.outcomeRes;
    // typical structure: outcomeRes = { risk: 'High Risk', probability: 0.72 }
    if (r.prediction && r.prediction.outcomeRes && typeof r.prediction.outcomeRes.probability === 'number'){
      const prob = r.prediction.outcomeRes.probability;
      const label = r.prediction.outcomeRes.risk === 'High Risk' ? 1 : 0;
      y_true.push(label);
      y_prob.push(prob);
      y_pred.push(prob >= 0.5 ? 1 : 0);
    }
  }

  if (y_true.length > 10){
    // send to server for ROC computation (server will run sklearn.roc_curve)
    const roc = await postJSON('/roc-data', { y_true, y_prob });
    plotROC(roc.fpr, roc.tpr, roc.auc);
    const cm = await postJSON('/confusion-matrix', { y_true, y_pred });
    renderConfMatrix(cm);
  } else {
    console.warn('Not enough labeled records to compute ROC/confusion (need >10)');
  }
}

function plotROC(fpr, tpr, aucVal){
  const ctx = document.getElementById('rocCurve').getContext('2d');
  if (rocChart) rocChart.destroy();
  rocChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: fpr,
      datasets: [{ label:`ROC (AUC=${(aucVal||0).toFixed(3)})`, data: tpr, fill:false }]
    },
    options: { responsive:true, scales:{ x:{ title:{display:true, text:'False Positive Rate'}}, y:{ title:{display:true, text:'True Positive Rate'}} } }
  });
}

function renderConfMatrix(cm){
  // cm is [[tn, fp],[fn, tp]] or similar
  const table = document.createElement('table');
  table.innerHTML = `<tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
    <tr><th>Actual 0</th><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr>
    <tr><th>Actual 1</th><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr>`;
  const container = document.getElementById('confMatrixTable');
  container.innerHTML = '';
  container.appendChild(table);
}

// INIT
window.addEventListener('load', async ()=>{
  firebase.auth().onAuthStateChanged(async user=>{
    if (!user) { window.location.href = 'login.html'; return; }
    document.getElementById('userName').innerText = user.displayName || user.email;
    await refreshStats();
    await loadFeatureImportance();
    // attempt ROC/confusion generation
    await generateModelMetricsFromRecords();
  });
});

// Chat open button
document.getElementById('openChat').addEventListener('click', ()=>{
  openChatModal();
});

// Basic chat modal (quick implementation)
function openChatModal(){
  // create modal if not exists
  if (document.getElementById('chatModal')) return showChat();
  const modal = document.createElement('div');
  modal.id = 'chatModal';
  modal.style.position='fixed'; modal.style.right='20px'; modal.style.bottom='20px';
  modal.style.width='320px'; modal.style.maxWidth='90%'; modal.style.zIndex=9999;
  modal.style.background='white'; modal.style.borderRadius='10px'; modal.style.boxShadow='0 8px 24px rgba(0,0,0,0.15)';
  modal.innerHTML = `<div style="padding:10px"><strong>AI Assistant</strong><div id="chatWindow" style="height:220px;overflow:auto;margin-top:8px;border:1px solid #eee;padding:8px;border-radius:6px"></div>
    <div style="display:flex;gap:8px;margin-top:8px">
      <input id="chatInput" style="flex:1;padding:8px;border:1px solid #ddd;border-radius:6px" placeholder="Ask clinical question..." />
      <button id="sendChat" class="primary">Send</button>
    </div></div>`;
  document.body.appendChild(modal);
  document.getElementById('sendChat').addEventListener('click', sendChatMessage);
}
function showChat(){ document.getElementById('chatModal').style.display='block' }
function appendChat(role, text){
  const w = document.getElementById('chatWindow');
  const el = document.createElement('div');
  el.style.margin='6px 0'; el.style.padding='8px'; el.style.borderRadius='8px';
  el.style.background = (role==='user') ? '#1d3557' : '#eef2ff';
  el.style.color = (role==='user') ? '#fff' : '#111';
  el.innerText = text;
  w.appendChild(el); w.scrollTop = w.scrollHeight;
}
async function sendChatMessage(){
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  appendChat('user', msg);
  input.value = '';
  appendChat('ai', 'Thinking...');
  try {
    const res = await postJSON('/api/chat', { message: msg });
    const reply = res.reply || res;
    // replace last 'Thinking...' bubble
    const w = document.getElementById('chatWindow');
    const last = w.lastChild;
    if (last) last.remove();
    appendChat('ai', reply);
  } catch(err){
    console.error('Chat error', err);
    appendChat('ai','Failed to get answer.');
  }
}

// ====================================================
// CHATBOT SYSTEM — SEND MESSAGE
// ====================================================
async function sendMessage() {
    const input = document.getElementById("chatInput");
    const message = input.value.trim();
    if (!message) return;

    appendMessage("You", message);
    saveChat("You", message);
    input.value = "";

    const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    });

    const data = await res.json();
    appendMessage("AI", data.reply);
    saveChat("AI", data.reply);
}


// ====================================================
// ADD MESSAGE TO CHATBOX + VOICE RESPONSE
// ====================================================
function appendMessage(sender, text) {
    const box = document.getElementById("chat-box");

    const div = document.createElement("div");
    div.classList.add("chat-line");
    div.innerHTML = `<strong>${sender}:</strong> ${text}`;

    box.appendChild(div);
    box.scrollTop = box.scrollHeight;

    if (sender === "AI") speak(text);
}


// ====================================================
// TEXT-TO-SPEECH (AI VOICE OUTPUT)
// ====================================================
function speak(text) {
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1;
    utter.pitch = 1;
    speechSynthesis.speak(utter);
}


// ====================================================
// VOICE INPUT (SPEECH RECOGNITION)
// ====================================================
let recognition;
if ("webkitSpeechRecognition" in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";

    recognition.onresult = function(event) {
        const text = event.results[0][0].transcript;
        document.getElementById("chatInput").value = text;
        sendMessage();
    };
}

function startVoice() {
    if (recognition) recognition.start();
}


// ====================================================
// SAVE CHAT HISTORY TO FIRESTORE
// ====================================================
async function saveChat(sender, text) {
    const user = auth.currentUser;
    if (!user) return;

    await firebase.firestore().collection("chat_history").add({
        uid: user.uid,
        sender,
        message: text,
        timestamp: firebase.firestore.FieldValue.serverTimestamp()
    });
}


// ====================================================
// GENERATE MEDICAL NOTE
// ====================================================
async function generateNote() {
    const chat = document.getElementById("chat-box").innerText;

    const res = await fetch(`${API_BASE}/generate-note`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ chat })
    });

    const data = await res.json();

    alert("Generated Structured Medical Note:\n\n" + data.note);
}
// =================================================
// SYMPTOM CHECKER, PATIENT PROFILE, DOCTOR ANALYTICS
// (Append this block to the end of dashboard.js)
// =================================================

/*
  Assumptions:
  - firebase initialized earlier in this file (firebase.initializeApp)
  - db = firebase.firestore() available
  - AppAuth.getUserRole(uid) exists (from auth.js)
  - API endpoints: /predict-disease and /predict-outcome exist on server
*/

const db = firebase.firestore(); // ensure db reference exists

// ---------- Symptom Checker ----------
document.getElementById('runSymptomCheck').addEventListener('click', async () => {
  const checkboxes = Array.from(document.querySelectorAll('.symptom'));
  const selected = checkboxes.filter(cb => cb.checked).map(cb => cb.value);

  if (selected.length === 0) {
    alert('Please select at least one symptom.');
    return;
  }

  // Build payload for predict endpoint: set boolean flags for each symptom
  const payload = {
    // minimal required shape for server preprocess
    Fever: selected.includes('Fever') ? 'Yes' : 'No',
    Cough: selected.includes('Cough') ? 'Yes' : 'No',
    Fatigue: selected.includes('Fatigue') ? 'Yes' : 'No',
    "Difficulty Breathing": selected.includes('Difficulty Breathing') ? 'Yes' : 'No',
    // additional placeholder features (age/gender) - better results if filled
    Age: Number(document.getElementById('age') ? document.getElementById('age').value : 30) || 30,
    Gender: document.getElementById('gender') ? document.getElementById('gender').value : 'Female',
    "Blood Pressure": document.getElementById('bp_cat') ? document.getElementById('bp_cat').value : 'Normal',
    "Cholesterol Level": document.getElementById('chol') ? document.getElementById('chol').value : 'Normal'
  };

  // call server predict-disease endpoint
  try {
    const res = await fetch('/predict-disease', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const box = document.getElementById('symptomResult');
    if (data.error) {
      box.innerText = 'Error: ' + data.error;
      box.style.display = 'block';
      return;
    }
    const tops = data.top3 || [];
    box.innerHTML = '<strong>Top suggestions:</strong><ul>' + tops.map(t => `<li>${t.disease} — ${(t.confidence*100).toFixed(1)}%</li>`).join('') + '</ul>';
    box.style.display = 'block';
  } catch (err) {
    console.error('Symptom check failed', err);
    alert('Symptom check failed: ' + err.message);
  }
});

// ---------- Patient Profile saving ----------
document.getElementById('savePatientBtn').addEventListener('click', async () => {
  const name = document.getElementById('pf_name').value.trim();
  const age = Number(document.getElementById('pf_age').value || 0);
  const gender = document.getElementById('pf_gender').value;
  const phone = document.getElementById('pf_phone').value.trim();
  const bp = document.getElementById('pf_bp').value;

  if (!name) { alert('Patient name required'); return; }

  try {
    const user = firebase.auth().currentUser;
    const doc = {
      name, age, gender, phone, blood_pressure: bp,
      createdBy: user ? user.uid : null,
      createdAt: firebase.firestore.FieldValue.serverTimestamp()
    };
    const ref = await db.collection('patients').add(doc);
    alert('Patient profile saved (id: ' + ref.id + ')');
    // optionally clear fields
    document.getElementById('pf_name').value = '';
    document.getElementById('pf_age').value = '';
    document.getElementById('pf_phone').value = '';
  } catch (err) {
    console.error('Save patient failed', err);
    alert('Save patient failed: ' + err.message);
  }
});

// ---------- Doctor Analytics (admin only) ----------
async function showAnalyticsIfAdmin() {
  const user = firebase.auth().currentUser;
  if (!user) return;
  let role = null;
  try {
    // get role from Firestore user doc
    const doc = await db.collection('users').doc(user.uid).get();
    if (doc.exists) role = doc.data().role;
  } catch (e) {
    console.warn('Failed to read user role', e);
  }

  if (role === 'admin') {
    document.getElementById('analyticsCard').style.display = 'block';
    await loadAnalyticsCharts();
  } else {
    document.getElementById('analyticsCard').style.display = 'none';
  }
}

async function loadAnalyticsCharts() {
  // query diagnosis records (last 12 months)
  const snap = await db.collection('diagnosis').orderBy('createdAt','desc').limit(1000).get();
  const docs = snap.docs.map(d => d.data());

  // timeseries (by day)
  const countsByDay = {};
  docs.forEach(r => {
    const ts = r.createdAt && r.createdAt.toDate ? r.createdAt.toDate() : (r.createdAt || new Date());
    const day = ts.toISOString().slice(0,10);
    countsByDay[day] = (countsByDay[day] || 0) + 1;
  });
  const days = Object.keys(countsByDay).sort();
  const counts = days.map(d => countsByDay[d]);

  // top diagnoses
  const dxCount = {};
  docs.forEach(r => {
    const top = (r.prediction && r.prediction.diseaseRes && r.prediction.diseaseRes.top3 && r.prediction.diseaseRes.top3[0] && r.prediction.diseaseRes.top3[0].disease)
                || (r.prediction && r.prediction.disease_top)
                || 'Unknown';
    dxCount[top] = (dxCount[top] || 0) + 1;
  });
  const dxLabels = Object.keys(dxCount).sort((a,b) => dxCount[b]-dxCount[a]).slice(0,8);
  const dxValues = dxLabels.map(l => dxCount[l]);

  // average risk (based on outcome probability)
  let riskSum = 0; let riskCount = 0;
  docs.forEach(r => {
    const prob = r.prediction && r.prediction.outcomeRes && r.prediction.outcomeRes.probability;
    if (typeof prob === 'number') { riskSum += prob; riskCount += 1; }
  });
  const avgRisk = riskCount ? (riskSum / riskCount) : null;
  document.getElementById('avgRisk').innerText = avgRisk !== null ? (avgRisk*100).toFixed(1) + '%' : 'N/A';

  // Render charts using Chart.js
  // Records time chart
  const ctx1 = document.getElementById('recordsTimeChart').getContext('2d');
  if (window._recordsTimeChart) window._recordsTimeChart.destroy();
  window._recordsTimeChart = new Chart(ctx1, {
    type: 'line',
    data: { labels: days, datasets: [{ label: 'Records/day', data: counts, fill:false }]},
    options: { responsive:true, scales:{ x:{ display:true }, y:{ beginAtZero:true } } }
  });

  // Top diagnoses chart
  const ctx2 = document.getElementById('topDxChart').getContext('2d');
  if (window._topDxChart) window._topDxChart.destroy();
  window._topDxChart = new Chart(ctx2, {
    type: 'bar',
    data: { labels: dxLabels, datasets: [{ label:'Count', data: dxValues }]},
    options: { responsive:true }
  });
}

// call admin check on load (ensure this runs after auth state resolved)
firebase.auth().onAuthStateChanged(user => {
  if (!user) return;
  showAnalyticsIfAdmin();
});

// optionally refresh charts periodically
setInterval(() => {
  firebase.auth().onAuthStateChanged(user => { if (user) showAnalyticsIfAdmin(); });
}, 1000 * 60 * 5); // every 5 minutes
