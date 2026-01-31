import { useState, useEffect } from 'react'
import { generate3d, status, downloadUrl, listJobs, deleteJob, listFolders, deleteFolder } from './api'
import { ModelViewer } from './ModelViewer'

const STATE_LABELS = {
  queued: 'Queued',
  generating_canonical: 'Canonical image',
  synthesizing_views: 'View synthesis',
  masking: 'Masking',
  reconstructing_3d: '3D reconstruction',
  post_processing: 'Post-processing',
  completed: 'Completed',
  failed: 'Failed',
}
function formatState(state) {
  return STATE_LABELS[state] || (state || '').replace(/_/g, ' ')
}

const LOADING_STATES = ['queued', 'generating_canonical', 'synthesizing_views', 'masking', 'reconstructing_3d', 'post_processing']

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [quality, setQuality] = useState('fast')
  const [jobId, setJobId] = useState(null)
  const [jobState, setJobState] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [glbUrl, setGlbUrl] = useState(null)
  const [history, setHistory] = useState([])
  const [viewingId, setViewingId] = useState(null)
  const [folders, setFolders] = useState({ input_folders: [], view_folders: [] })
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [loadingFolders, setLoadingFolders] = useState(true)
  const [deletingJobId, setDeletingJobId] = useState(null)
  const [deletingFolder, setDeletingFolder] = useState(null)

  const loadHistory = async () => {
    setLoadingHistory(true)
    try {
      const { jobs } = await listJobs()
      setHistory(jobs || [])
    } catch {
      setHistory([])
    } finally {
      setLoadingHistory(false)
    }
  }

  const loadFolders = async () => {
    setLoadingFolders(true)
    try {
      const data = await listFolders()
      setFolders(data)
    } catch {
      setFolders({ input_folders: [], view_folders: [] })
    } finally {
      setLoadingFolders(false)
    }
  }

  useEffect(() => {
    loadHistory()
    loadFolders()
  }, [])

  const pollStatus = async (id) => {
    const data = await status(id)
    setJobState(data.state)
    if (data.state === 'completed' && data.asset_path) {
      setGlbUrl(downloadUrl(id))
      setViewingId(id)
      loadHistory()
      loadFolders()
      return true
    }
    if (data.state === 'failed') {
      setError(data.message || 'Job failed')
      loadHistory()
      loadFolders()
      return true
    }
    return false
  }

  const isJobWaiting = jobId && jobState && LOADING_STATES.includes(jobState)
  const isAnyAction = loading || loadingHistory || loadingFolders || deletingJobId || deletingFolder || isJobWaiting

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setJobId(null)
    setJobState(null)
    setGlbUrl(null)
    setViewingId(null)
    if (!prompt.trim()) return
    setLoading(true)
    try {
      const { job_id } = await generate3d({
        prompt: prompt.trim(),
        output_format: 'glb',
        quality_tier: quality,
      })
      setJobId(job_id)
      setJobState('queued')
      const poll = async () => {
        const done = await pollStatus(job_id)
        if (done) setLoading(false)
        else setTimeout(poll, 2000)
      }
      setTimeout(poll, 1500)
    } catch (err) {
      setError(err.message || 'Request failed')
      setLoading(false)
    }
  }

  const handleView = (id) => {
    const job = history.find((j) => j.job_id === id)
    if (job?.state === 'completed') {
      setGlbUrl(downloadUrl(id))
      setViewingId(id)
    }
  }

  const handleDelete = async (id) => {
    setDeletingJobId(id)
    try {
      await deleteJob(id)
      if (viewingId === id) {
        setGlbUrl(null)
        setViewingId(null)
      }
      await loadHistory()
      await loadFolders()
    } catch (err) {
      setError(err.message || 'Delete failed')
    } finally {
      setDeletingJobId(null)
    }
  }

  const handleDeleteFolder = async (type, name) => {
    const key = `${type}/${name}`
    setDeletingFolder(key)
    try {
      await deleteFolder(type, name)
      await loadFolders()
    } catch (err) {
      setError(err.message || 'Delete folder failed')
    } finally {
      setDeletingFolder(null)
    }
  }

  const formatFolderDate = (mtime) => {
    if (!mtime) return '—'
    return new Date(mtime * 1000).toLocaleString()
  }

  const formatDate = (ts) => {
    if (!ts) return '—'
    const d = new Date(parseInt(ts, 10) * 1000)
    return d.toLocaleString()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-slate-100 font-sans">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <header className="text-center mb-14">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight bg-gradient-to-r from-cyan-300 via-violet-300 to-fuchsia-300 bg-clip-text text-transparent">
            3D Generate
          </h1>
          <p className="mt-3 text-slate-400 text-lg">
            Prompt → FLUX images → mesh. Served at{' '}
            <a
              href="https://elohim-bitch-gpu.insanelabs.org"
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:underline"
            >
              elohim-bitch-gpu.insanelabs.org
            </a>
          </p>
        </header>

        <form
          onSubmit={handleSubmit}
          className="rounded-2xl bg-slate-800/60 border border-slate-700/80 shadow-xl shadow-black/20 p-6 md:p-8 mb-10"
        >
          <label className="block text-sm font-medium text-slate-300 mb-2">Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g. a red ceramic vase on neutral gray background"
            rows={3}
            className="w-full rounded-xl bg-slate-900/80 border border-slate-600 text-slate-100 placeholder-slate-500 px-4 py-3 focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500 outline-none transition"
            disabled={isAnyAction}
          />
          <div className="mt-4 flex flex-wrap items-center gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Quality / speed</label>
              <select
                value={quality}
                onChange={(e) => setQuality(e.target.value)}
                className="rounded-lg bg-slate-900/80 border border-slate-600 text-slate-100 px-3 py-2 focus:ring-2 focus:ring-cyan-500/50 outline-none"
                disabled={isAnyAction}
              >
                <option value="fast">Fast (2 views)</option>
                <option value="standard">Standard (4 views)</option>
                <option value="high">High (8 views)</option>
              </select>
            </div>
            <button
              type="submit"
              disabled={isAnyAction || !prompt.trim()}
              className="mt-5 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-violet-600 hover:from-cyan-500 hover:to-violet-500 disabled:opacity-50 disabled:pointer-events-none font-semibold text-white shadow-lg shadow-cyan-500/20 transition"
            >
              {loading || isJobWaiting ? 'Generating…' : 'Generate 3D'}
            </button>
          </div>
        </form>

        {isAnyAction && (
          <div className="fixed inset-0 bg-slate-950/70 backdrop-blur-sm flex items-center justify-center z-50 pointer-events-none">
            <div className="rounded-2xl bg-slate-800 border border-slate-600 px-8 py-6 flex items-center gap-4 shadow-xl">
              <svg className="animate-spin h-8 w-8 text-cyan-400" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span className="text-slate-200 font-medium">
                {loading || isJobWaiting ? 'Generating…' : loadingHistory ? 'Loading jobs…' : loadingFolders ? 'Loading folders…' : deletingJobId ? 'Deleting job…' : deletingFolder ? 'Deleting folder…' : 'Please wait…'}
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-xl bg-red-950/50 border border-red-800 text-red-200 px-4 py-3 mb-8">
            {error}
          </div>
        )}

        {jobId && (
          <div className="rounded-2xl bg-slate-800/40 border border-slate-700/60 p-6 mb-10">
            <p className="text-slate-400 text-sm font-mono mb-1">Job ID</p>
            <p className="text-cyan-300 font-mono break-all">{jobId}</p>
            <p className="mt-2 text-slate-400">
              Status: <span className="text-slate-200">{formatState(jobState)}</span>
            </p>
          </div>
        )}

        {(glbUrl || viewingId) && (
          <section className="rounded-2xl bg-slate-800/40 border border-slate-700/60 overflow-hidden mb-10">
            <div className="px-6 py-4 border-b border-slate-700/60 flex items-center justify-between flex-wrap gap-2">
              <h2 className="text-xl font-semibold text-slate-200">3D Preview</h2>
              {glbUrl && (
                <a
                  href={glbUrl}
                  download
                  className="text-sm text-cyan-400 hover:underline"
                >
                  Download GLB
                </a>
              )}
            </div>
            <div className="aspect-square min-h-[320px] bg-slate-900/80">
              {glbUrl ? (
                <ModelViewer src={glbUrl} />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-slate-500">
                  Select a completed job below to preview
                </div>
              )}
            </div>
          </section>
        )}

        <section className="rounded-2xl bg-slate-800/40 border border-slate-700/60 overflow-hidden mb-10">
          <div className="px-6 py-4 border-b border-slate-700/60">
            <h2 className="text-xl font-semibold text-slate-200">Past generations</h2>
            <p className="text-slate-400 text-sm mt-1">Click to view, or delete to remove</p>
          </div>
          <ul className="divide-y divide-slate-700/60 max-h-[320px] overflow-y-auto">
            {loadingHistory && history.length === 0 && (
              <li className="px-6 py-8 text-slate-500 text-center">Loading jobs…</li>
            )}
            {!loadingHistory && history.length === 0 && (
              <li className="px-6 py-8 text-slate-500 text-center">No generations yet</li>
            )}
            {history.map((job) => (
              <li
                key={job.job_id}
                className="px-6 py-4 flex items-center justify-between gap-4 hover:bg-slate-700/30"
              >
                <button
                  type="button"
                  onClick={() => handleView(job.job_id)}
                  disabled={job.state !== 'completed'}
                  className="flex-1 text-left min-w-0"
                >
                  <p className="text-slate-200 truncate" title={job.prompt || job.job_id}>
                    {job.prompt || job.job_id}
                  </p>
                  <p className="text-slate-500 text-sm mt-0.5">
                    {formatDate(job.created_at)} · {formatState(job.state)}
                  </p>
                </button>
                <div className="flex items-center gap-2 shrink-0">
                  {job.state === 'completed' && (
                    <a
                      href={downloadUrl(job.job_id)}
                      download
                      className="text-sm text-cyan-400 hover:underline"
                    >
                      Download
                    </a>
                  )}
                  <button
                    type="button"
                    onClick={() => handleDelete(job.job_id)}
                    disabled={deletingJobId === job.job_id}
                    className="px-3 py-1.5 rounded-lg bg-red-900/50 text-red-200 hover:bg-red-800/50 text-sm disabled:opacity-50"
                  >
                    {deletingJobId === job.job_id ? '…' : 'Delete'}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section className="rounded-2xl bg-slate-800/40 border border-slate-700/60 overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-700/60">
            <h2 className="text-xl font-semibold text-slate-200">Generated folders</h2>
            <p className="text-slate-400 text-sm mt-1">Input and view-synthesis folders on disk. Delete to free space.</p>
          </div>
          <div className="p-4 max-h-[360px] overflow-y-auto">
            {loadingFolders && folders.input_folders.length === 0 && folders.view_folders.length === 0 && (
              <p className="text-slate-500 text-center py-6">Loading folders…</p>
            )}
            {!loadingFolders && (
              <>
                {folders.input_folders.length > 0 && (
                  <div className="mb-4">
                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-2">Inputs</p>
                    <ul className="space-y-1">
                      {folders.input_folders.map((f) => (
                        <li key={`input-${f.name}`} className="flex items-center justify-between gap-2 py-1.5 px-2 rounded-lg hover:bg-slate-700/30">
                          <span className="text-slate-300 font-mono text-sm truncate" title={f.name}>{f.name}</span>
                          <span className="text-slate-500 text-xs shrink-0">{formatFolderDate(f.mtime)}</span>
                          <button
                            type="button"
                            onClick={() => handleDeleteFolder('input', f.name)}
                            disabled={deletingFolder === `input/${f.name}`}
                            className="px-2 py-1 rounded bg-red-900/50 text-red-200 text-xs hover:bg-red-800/50 disabled:opacity-50"
                          >
                            {deletingFolder === `input/${f.name}` ? '…' : 'Delete'}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {folders.view_folders.length > 0 && (
                  <div>
                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-2">Views</p>
                    <ul className="space-y-1">
                      {folders.view_folders.map((f) => (
                        <li key={`view-${f.name}`} className="flex items-center justify-between gap-2 py-1.5 px-2 rounded-lg hover:bg-slate-700/30">
                          <span className="text-slate-300 font-mono text-sm truncate" title={f.name}>{f.name}</span>
                          <span className="text-slate-500 text-xs shrink-0">{formatFolderDate(f.mtime)}</span>
                          <button
                            type="button"
                            onClick={() => handleDeleteFolder('view', f.name)}
                            disabled={deletingFolder === `view/${f.name}`}
                            className="px-2 py-1 rounded bg-red-900/50 text-red-200 text-xs hover:bg-red-800/50 disabled:opacity-50"
                          >
                            {deletingFolder === `view/${f.name}` ? '…' : 'Delete'}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {folders.input_folders.length === 0 && folders.view_folders.length === 0 && (
                  <p className="text-slate-500 text-center py-6">No folders yet</p>
                )}
              </>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
