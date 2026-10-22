import { useCallback, useEffect, useState } from "react";

/** Audio input devices. Labels are blank until the user has granted mic permission. */
export function useDevices(): { devices: MediaDeviceInfo[]; error: string | null; refresh(): void } {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setError("This browser cannot list audio devices.");
      return;
    }
    navigator.mediaDevices
      .enumerateDevices()
      .then((list) => {
        setDevices(list.filter((d) => d.kind === "audioinput"));
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Could not list audio devices."));
  }, []);

  useEffect(() => {
    refresh();
    const md = navigator.mediaDevices;
    if (!md?.addEventListener) return;
    md.addEventListener("devicechange", refresh);
    return () => md.removeEventListener("devicechange", refresh);
  }, [refresh]);

  return { devices, error, refresh };
}
