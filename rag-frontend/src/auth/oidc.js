import { UserManager, WebStorageStateStore } from 'oidc-client-ts'

const runtimeConfig = globalThis.window?.__RAG_CONFIG__ || {}
const issuer = String(
  runtimeConfig.oidcIssuer || import.meta.env?.VITE_OIDC_ISSUER || ''
).replace(/\/$/, '')
const clientId = String(runtimeConfig.oidcClientId || import.meta.env?.VITE_OIDC_CLIENT_ID || '')
const audience = String(
  runtimeConfig.oidcAudience || import.meta.env?.VITE_OIDC_AUDIENCE || 'rag-api'
)

export const oidcEnabled = Boolean(issuer && clientId)

const userManager = oidcEnabled
  ? new UserManager({
      authority: issuer,
      client_id: clientId,
      redirect_uri: `${window.location.origin}/`,
      post_logout_redirect_uri: `${window.location.origin}/`,
      response_type: 'code',
      scope: 'openid profile email',
      extraQueryParams: { audience },
      userStore: new WebStorageStateStore({ store: window.sessionStorage }),
      automaticSilentRenew: false,
      monitorSession: true
    })
  : null

let authenticatedUser = null

export const parseSigninCallback = (search = '') => {
  const params = new URLSearchParams(search)
  if (!params.has('state')) return null
  if (params.has('error')) {
    return {
      kind: 'error',
      error: params.get('error') || 'unknown_error',
      description: params.get('error_description') || ''
    }
  }
  return params.has('code') ? { kind: 'success' } : null
}

const clearSigninCallback = () => {
  window.history.replaceState(
    {},
    document.title,
    `${window.location.pathname}${window.location.hash}`
  )
}

export const initializeAuth = async () => {
  if (!userManager) return null

  const callback = parseSigninCallback(window.location.search)
  if (callback?.kind === 'error') {
    clearSigninCallback()
    const description = callback.description ? `: ${callback.description}` : ''
    throw new Error(`OIDC sign-in failed (${callback.error})${description}`)
  }

  if (callback?.kind === 'success') {
    authenticatedUser = await userManager.signinRedirectCallback()
    clearSigninCallback()
  } else {
    authenticatedUser = await userManager.getUser()
  }

  if (!authenticatedUser || authenticatedUser.expired) {
    await userManager.signinRedirect()
    return null
  }

  userManager.events.addAccessTokenExpired(() => {
    void userManager.signinRedirect()
  })
  return authenticatedUser
}

export const getAuthenticatedUser = () => authenticatedUser

export const getAccessToken = async () => {
  if (!userManager) return null
  const user = await userManager.getUser()
  if (!user || user.expired) {
    await userManager.signinRedirect()
    return null
  }
  authenticatedUser = user
  return user.access_token
}

export const authorizedFetch = async (input, init = {}) => {
  const token = await getAccessToken()
  const headers = new Headers(init.headers || {})
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(input, { ...init, headers })
}

export const signOut = async () => {
  if (!userManager) return
  await userManager.signoutRedirect()
}
