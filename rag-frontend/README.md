# RAG frontend

## Development

```powershell
npm install
npm run dev
```

Vite proxies `/api` and `/health` to `RAG_API_UPSTREAM`, which defaults to
`http://127.0.0.1:8000`. Browser requests therefore remain same-origin.

## OIDC authentication

The browser uses OIDC Authorization Code + PKCE through `oidc-client-ts`.
Provide these values at build time:

```dotenv
VITE_OIDC_ISSUER=http://localhost:18080/realms/industrial-rag
VITE_OIDC_CLIENT_ID=rag-frontend
VITE_OIDC_AUDIENCE=rag-api
```

When issuer or client ID is empty, authentication is disabled for laptop
development. When enabled, startup redirects unauthenticated users to the IdP
and API requests carry an `Authorization: Bearer` header. Never place a
permanent API key in a `VITE_*` variable.

Set `VITE_API_BASE` only when the API intentionally uses another public
origin. The backend CORS allow-list must explicitly include that origin.

The included container implements this contract. From the repository root:

```powershell
docker compose --profile frontend up -d --build
```

It reads `RAG_API_UPSTREAM` and `OIDC_CONNECT_SRC` at container startup. Nginx
proxies `/api/*` but does not inject a shared administrator credential.
