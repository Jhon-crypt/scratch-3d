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
      <div className="w-full h-full min-h-[320px] flex items-center justify-center text-slate-500 bg-slate-900/80">
        Could not load 3D model. Try downloading the GLB.
      </div>
    )
  }

  return <div ref={containerRef} className="w-full h-full min-h-[320px]" />
}
