import FIREBASE_CONFIG from './firebase-config.js';

if (!window.firebase) throw new Error('Firebase SDK missing');
const app = firebase.apps.length ? firebase.app() : firebase.initializeApp(FIREBASE_CONFIG);
const auth = app.auth();
const db = app.firestore();

async function register(email, password, displayName) {
  const cred = await auth.createUserWithEmailAndPassword(email, password);
  await cred.user.updateProfile({displayName});
  await db.collection('users').doc(cred.user.uid).set({
    email, displayName, role: 'doctor',
    createdAt: firebase.firestore.FieldValue.serverTimestamp()
  });
  return cred.user;
}
const login = (email, password) => auth.signInWithEmailAndPassword(email, password);
async function getUserRole(uid) {
  const snapshot = await db.collection('users').doc(uid).get();
  return snapshot.exists ? snapshot.data().role : null;
}
window.AppAuth = {register, login, getUserRole, auth, db};
export default window.AppAuth;
