import FIREBASE_CONFIG from './firebase-config.js';

if (!window.firebase) throw new Error('Firebase SDK missing - load firebase-app and firebase-auth scripts');
firebase.initializeApp(FIREBASE_CONFIG);
const auth = firebase.auth();
const db = firebase.firestore();

/**
 * register(email, password, displayName, role)
 * - creates the user and writes a users/{uid} doc with role
 */
async function register(email, password, displayName, role='doctor'){
  const cred = await auth.createUserWithEmailAndPassword(email, password);
  await cred.user.updateProfile({ displayName });
  await db.collection('users').doc(cred.user.uid).set({
    email, displayName, role, createdAt: firebase.firestore.FieldValue.serverTimestamp()
  });
  return cred.user;
}

function login(email,password){
  return auth.signInWithEmailAndPassword(email,password);
}

async function getUserRole(uid){
  const doc = await db.collection('users').doc(uid).get();
  if (!doc.exists) return null;
  return doc.data().role;
}

window.AppAuth = { register, login, getUserRole, auth, db };
export default window.AppAuth;
