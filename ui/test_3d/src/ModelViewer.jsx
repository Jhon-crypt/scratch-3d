import { useEffect, useRef, useState } from 'react'

export function ModelViewer({ src }) {
  const containerRef = useRef(null)
  const [loadError, setLoadError] = useState(false)

  useEffect(() => {
    if (!src || !containerRef.current) return
    setLoadError(false)
    const el = document.createElement('model-viewer')
    el.setAttribute('src', src)
    el.setAttribute('alt', '3D model')
    el.setAttribute('auto-rotate', '')
    el.setAttribute('camera-controls', '')
    el.setAttribute('shadow-intensity', '1')
    el.setAttribute('shadow-softness', '0.8')
    el.setAttribute('exposure', '1.15')
    el.setAttribute('tone-mapping', 'commerce')
    el.setAttribute('camera-orbit', '45deg 75deg 2.2m')
    el.setAttribute('min-camera-orbit', 'auto auto 1.2m')
    el.setAttribute('max-camera-orbit', 'auto auto 6m')
    // Use default lighting (external HDR URLs like modelviewer.dev can 404)
    el.setAttribute('style', 'width:100%;height:100%;min-height:320px;background:#0f172a')
    el.addEventListener('error', () => setLoadError(true))
    containerRef.current.innerHTML = ''
    containerRef.current.appendChild(el)
    return () => {
      el.removeEventListener('error', () => setLoadError(true))
      containerRef.current?.removeChild(el)
    }
  }, [src])

  if (loadError) {
    return (
      <div className="w-full h-full min-h-[320px] flex flex-col items-center justify-center gap-3 text-slate-400 bg-slate-900/80 p-4">
        <p>Could not load 3D model in viewer.</p>
        {src && (
          <a
            href={src}
            download
            className="text-cyan-400 hover:underline font-medium"
          >
            Download GLB
          </a>
        )}
      </div>
    )
  }

  return <div ref={containerRef} className="w-full h-full min-h-[320px]" />
}
