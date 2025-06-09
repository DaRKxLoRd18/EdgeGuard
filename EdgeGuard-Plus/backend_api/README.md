# EdgeGuard++ Backend (Node.js + MongoDB)
This is the Express.js backend for receiving encrypted alert metadata and storing it in MongoDB.

## Endpoints
- POST `/api/alerts` — Submit alert JSON (including IV and ciphertext)
