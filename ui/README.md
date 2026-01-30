# UI — 3D Test

React + Vite + Tailwind app for testing the 3D pipeline at **/3d-test**.

## Run automatically (Docker)

From repo root: **`docker-compose up -d`** — the **ui** service builds and serves the app; host nginx proxies **/3d-test/** to it. No Node on the host required.

## Domain

- **Production**: App should be served at `https://elohim-bitch-gpu.insanelabs.org/3d-test/`.
- **API**: Requests go to `elohim-bitch-gpu.insanelabs.org` (or `VITE_API_BASE` in `.env`).

## Stack

- [Vite](https://vite.dev/) + [React](https://react.dev/)
- [Tailwind CSS v4](https://tailwindcss.com/docs/installation/using-vite) via `@tailwindcss/vite`
- [model-viewer](https://modelviewer.dev/) for GLB display
