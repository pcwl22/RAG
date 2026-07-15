# RAG frontend

## Development

```powershell
npm install
npm run dev
```

Vite proxies `/api` and `/health` to `RAG_API_UPSTREAM`, which defaults to
`http://127.0.0.1:8000`. Browser requests therefore remain same-origin.

## Production authentication

Serve the built frontend behind a reverse proxy or BFF using the same paths.
The proxy should forward `/api/*` to the RAG API and inject `X-API-Key` from a
server-side secret when API-key authentication is enabled. Do not place the
key in a `VITE_*` variable because Vite variables are public browser code.

Set `VITE_API_BASE` only when the API intentionally uses another public
origin. The backend CORS allow-list must explicitly include that origin.

The included container implements this contract. From the repository root:

```powershell
docker compose --profile frontend up -d --build
```

It reads `RAG_API_UPSTREAM` and `RAG_API_KEY` only at container startup and
injects the key into proxied `/api/*` requests. Neither value is bundled into
the frontend JavaScript.
