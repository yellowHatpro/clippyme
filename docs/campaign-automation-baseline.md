# Campaign automation baseline

Recorded on 2026-09-28 before Phase 1 implementation, from upstream commit
`9959228` on the `campaign-automation` branch.

## Backend

- `pytest -m "not integration" -q`: **1147 passed, 1 skipped, 11 deselected**.
- `ruff check src/clippyme tests --select E9,F63,F7,F82`: **passed**.
- Warnings were limited to dependency deprecations (Starlette/httpx and
  google-genai on the host's Python 3.14).

## Dashboard

- `npm ci`: **passed** after normalizing ownership of the bind-mounted
  `dashboard/node_modules` directory left by Docker.
- `npm run lint`: **passed**.
- `npm run build`: **passed**.
- Host `npm test` under Node **26.8.1**: **181 passed, 9 failed**. All nine
  failures came from `localStorage` being unavailable unless Node is given a
  `--localstorage-file`; affected suites were `storage.test.js`,
  `useSessionPersistence.test.jsx`, and `useClipStates.test.jsx`. This is an
  environment/runtime issue rather than a ClippyMe source regression. The
  same suite passes in the supported Docker Node 24 environment.

## Docker

- `docker compose up --build`: images built and both services started.
- Backend and frontend health checks became healthy.

No unrelated upstream behavior was changed to make the host Node 26-only test
failure disappear.
