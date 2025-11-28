Firebase setup instructions (summary)

1. Create a Firebase project at console.firebase.google.com.

2. Authentication:
   - Build -> Authentication -> Get started.
   - Enable Email/Password.
   - (Optional) Enable Google sign-in.

3. Web app:
   - Project Settings -> General -> Add app (Web).
   - Copy the SDK config and paste into frontend/firebase-config.js.

4. Firestore:
   - Build -> Firestore Database -> Create database (start in test mode then lock down rules).
   - Paste the contents of firestore-rules.txt into Firestore rules tab.

5. Admin users:
   - Use Firebase Admin SDK to set custom claims for admin:
     Example (Node):
       const admin = require('firebase-admin');
       admin.auth().setCustomUserClaims(uid, { role: 'admin' });

6. Service account (server-side writes):
   - Project Settings -> Service accounts -> Generate new private key -> download JSON.
   - Upload JSON to your backend host (do not commit to git).
   - Set env var on host: GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccount.json

7. Test login -> create records -> verify Firestore docs.

8. Security:
   - Do not store PHI in unsecured locations.
   - Use HTTPS and environment variables for secrets.
