const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (import.meta.env.DEV ? '/3d-test-api' : 'https://elohim-bitch-gpu.insanelabs.org')

export async function health() {
  const r = await fetch(`${API_BASE}/health`)
  if (!r.ok) throw new Error(r.statusText)
  return r.json()
}

export async function generate3d({ prompt, output_format = 'glb', quality_tier = 'standard' }) {
  const r = await fetch(`${API_BASE}/3d/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, output_format, quality_tier }),
  })
  if (!r.ok) throw new Error((await r.text()) || r.statusText)
  return r.json()
}

export async function status(jobId) {
  const r = await fetch(`${API_BASE}/3d/status/${jobId}`)
  if (!r.ok) throw new Error(r.statusText)
  return r.json()
}

export function downloadUrl(jobId) {
  return `${API_BASE}/3d/download/${jobId}`
}

export async function listJobs() {
  const r = await fetch(`${API_BASE}/3d/jobs`)
  if (!r.ok) throw new Error(r.statusText)
  return r.json()
}

export async function deleteJob(jobId) {
  const r = await fetch(`${API_BASE}/3d/job/${jobId}`, { method: 'DELETE' })
  if (!r.ok) throw new Error(r.statusText)
  return r.json()
}
