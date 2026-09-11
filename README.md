# CliniCore Intelligence

An AI-assisted clinical decision-support **research prototype** by Ebenezer Kwaw.

> **MVP testing only:** This software is not a medical device, does not provide a diagnosis, and must not be used as the sole basis for treatment or emergency decisions. Use synthetic or properly de-identified data during testing.

## Current MVP

- Firebase email/password authentication
- Doctor test accounts with protected Firestore records
- XGBoost condition-candidate and outcome-risk models
- Top-three model outputs with confidence values
- Patient test-profile and prediction logging
- Feature-importance chart
- Clinician-facing AI information assistant
- Draft note generation for clinician verification
- Flask API, Docker deployment, and health endpoint

## Important limitations

The bundled dataset and models are suitable for software demonstration and controlled evaluation only. They have not been externally validated for clinical use. Confidence values are model scores, not the probability that a patient has a condition. Do not enter real patient identifiers into a public test deployment.

## Run locally

1. Copy your Firebase web configuration into `Frontend/firebase-config.js`.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

3. Optionally set AI configuration:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
```

4. Start the server:

```bash
python backend/server.py
```

Open `http://localhost:5000`. Check model readiness at `http://localhost:5000/health`.

## Docker

```bash
docker build -t clinicore-intelligence .
docker run --rm -p 5000:5000 -e OPENAI_API_KEY="your-key" clinicore-intelligence
```

## Firebase setup

- Enable Email/Password authentication and Firestore.
- Deploy `firebase/firebase-rule.txt` as the Firestore ruleset.
- New public registrations are always assigned the `doctor` role.
- Create administrator custom claims only from a trusted server-side process.
- Use a separate Firebase project containing no real clinical data for public MVP testing.

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Service and model readiness |
| `/predict-disease` | POST | Top-three model condition candidates |
| `/predict-outcome` | POST | Model-estimated outcome risk |
| `/feature-importance` | GET | Global model feature importance |
| `/chat` | POST | Clinician-facing information assistant |
| `/generate-note` | POST | Draft note requiring verification |

## Deployment variables

- `PORT`: provided by most hosting platforms
- `OPENAI_API_KEY`: optional; required for chat and note generation
- `OPENAI_MODEL`: optional model override
- `ALLOWED_ORIGINS`: comma-separated origins for split frontend/backend deployment
- `FLASK_DEBUG=1`: local development only

## MVP acceptance checks

- `GET /health` reports both models loaded.
- Registration creates a doctor account; users cannot self-register as admin.
- Invalid ages and malformed JSON return safe 400 responses.
- Prediction results show the prototype disclaimer.
- A doctor can read only their own Firestore records.
- Chat fails safely when no API key is configured.
- Generated notes are clearly marked as drafts requiring clinician review.

## License

See [LICENSE](LICENSE).
