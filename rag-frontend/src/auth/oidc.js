import { UserManager, WebStorageStateStore } from 'oidc-client-ts'

const issuer = import.meta.env?.VITE_OIDC_ISSUER?.replace(/\/$/, '') || ''
const clientId = import.meta.env?.VITE_OIDC_CLIENT_ID || ''
const audience = import.meta.env?.VITE_OIDC_AUDIENCE || 'rag-api'

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

const isSigninCallback = () => {
  const params = new URLSearchParams(window.location.search)
  return params.has('code') && params.has('state')
}

export const initializeAuth = async () => {
  if (!userManager) return null

  if (isSigninCallback()) {
    authenticatedUser = await userManager.signinRedirectCallback()
    window.history.replaceState({}, document.title, `${window.location.pathname}${window.location.hash}`)
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
