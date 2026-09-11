# CliniCore Intelligence MVP Pilot Test Plan

## Purpose

Evaluate usability, technical reliability, output clarity, and safety in a controlled prototype environment. This pilot does not evaluate clinical effectiveness and does not authorize clinical use.

## Participants

Target 5 to 10 invited reviewers:

- Clinicians or supervised health professionals
- Public health or health informatics reviewers
- Software or machine learning reviewers

Participants must be told that the system is a research prototype and not a medical device.

## Data rules

- Use synthetic scenarios or properly de-identified records only.
- Do not enter names, phone numbers, addresses, medical record numbers, dates of birth, or other patient identifiers.
- Do not use the platform during emergencies or for real treatment decisions.
- Delete pilot records after the evaluation window.

## Test workflow

1. Confirm the service status endpoint reports both models loaded.
2. Register a doctor test account.
3. Run the five standard synthetic scenarios in `docs/pilot_test_cases.json`.
4. Record whether each page and feature works.
5. Review whether the model output is understandable and appropriately qualified.
6. Test chatbot and draft-note behavior without including personal information.
7. Submit defects and safety concerns using the pilot feedback issue form.
8. Stop testing and flag the output immediately if it appears unsafe or misleading.

## Acceptance criteria

| Area | MVP acceptance target |
| --- | --- |
| Availability | Health endpoint succeeds before each session |
| API reliability | All automated tests pass |
| Access control | Users cannot self-register as admin |
| Data separation | Doctor reads only records created by that account |
| Validation | Invalid ages and malformed requests are rejected |
| Safety | Every output is visibly labelled as a prototype |
| AI assistant | Missing API configuration fails safely |
| Notes | Generated notes are marked as drafts requiring review |
| Usability | At least 80% of planned tasks completed without assistance |
| Critical defects | Zero unresolved security or high-severity safety defects |

## Feedback categories

- Bug
- Usability
- Incorrect or unclear model output
- Unsafe or misleading output
- Privacy or access-control concern
- Feature request

Do not paste patient data, credentials, API keys, Firebase configuration secrets, or full chat transcripts into GitHub issues.

## Go or no-go decision

Proceed to a broader demonstration only if automated checks pass, no critical security or safety issue remains open, Firebase rules have been deployed and verified, and the project owner has reviewed the pilot findings.

Even after a successful MVP pilot, the system remains a non-clinical research prototype until appropriate dataset review, external validation, governance, regulatory assessment, and prospective evaluation are completed.
