// Generic small helper functions used across pages:
export function q(sel){return document.querySelector(sel)}
export function on(sel,ev,fn){document.querySelector(sel).addEventListener(ev,fn)}
export const API_BASE = '/api' // adjust if backend host differs
