# Firebase Setup Guide (short)

1. Create a Firebase project at https://console.firebase.google.com/
2. Enable Authentication -> Email/Password
3. Create Firestore database (in production mode) and enable required collections
4. Create a service account (Project Settings -> Service accounts) and download the JSON.
5. In your backend, initialize the Admin SDK:

   ```python
   import firebase_admin
   from firebase_admin import credentials, auth
   cred = credentials.Certificate('/path/to/serviceAccountKey.json')
   firebase_admin.initialize_app(cred)
   ```
6. Use `auth.set_custom_user_claims(uid, {'role':'admin'})` to add roles.
7. In the client, after sign-in, fetch the ID token and decode claims to control UI.
8. Secure the claim-setting endpoint: allow only authenticated service accounts or restrict via IAM.
