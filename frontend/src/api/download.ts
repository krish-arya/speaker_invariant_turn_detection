/** Save an export response to disk using the server's Content-Disposition filename. */

export function filenameFromContentDisposition(header: string | null, fallback: string): string {
  if (!header) return fallback;
  const star = /filename\*\s*=\s*(?:UTF-8'')?([^;]+)/i.exec(header);
  if (star?.[1]) {
    try {
      return sanitize(decodeURIComponent(star[1].trim().replace(/^"|"$/g, "")), fallback);
    } catch {
      // fall through to the plain filename parameter
    }
  }
  const plain = /filename\s*=\s*("?)([^";]+)\1/i.exec(header);
  return plain?.[2] ? sanitize(plain[2].trim(), fallback) : fallback;
}

function sanitize(name: string, fallback: string): string {
  const cleaned = name.replace(/[\\/:*?"<>|\u0000-\u001f]/g, "_").trim();
  return cleaned || fallback;
}

export async function downloadResponse(res: Response, fallbackName: string): Promise<string> {
  const filename = filenameFromContentDisposition(res.headers.get("Content-Disposition"), fallbackName);
  const blob = await res.blob();
  saveBlob(blob, filename);
  return filename;
}

export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the browser a moment to start the download before revoking.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
