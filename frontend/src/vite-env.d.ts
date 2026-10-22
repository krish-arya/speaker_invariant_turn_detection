/// <reference types="vite/client" />

declare const __APP_VERSION__: string;
declare const __APP_BUILD_TIME__: string;

interface ImportMetaEnv {
  /** Optional absolute backend origin, e.g. http://localhost:8000. Empty = same origin (Vite proxy). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
