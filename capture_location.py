"""Nonblocking Windows location updates and filesystem-safe capture names."""

from datetime import datetime
import math
import os
import subprocess
import threading
import time


# Windows PowerShell supplies the built-in .NET location API; no pip dependency.
LOCATION_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Device
$watcher = New-Object System.Device.Location.GeoCoordinateWatcher
$culture = [Globalization.CultureInfo]::InvariantCulture
$ipCoord = $null
$ipTime = 0
try {
    $watcher.Start()
    while ($true) {
        $position = $watcher.Position
        $now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        if ($watcher.Permission -ne 'Denied' -and $watcher.Status -eq 'Ready' -and -not $position.Location.IsUnknown) {
            $lat = $position.Location.Latitude.ToString('R', $culture)
            $lon = $position.Location.Longitude.ToString('R', $culture)
            $stamp = $position.Timestamp.ToUnixTimeMilliseconds().ToString($culture)
            [Console]::WriteLine("$lat|$lon|$stamp")
        } elseif ($watcher.Permission -ne 'Denied') {
            if ($null -eq $ipCoord -or ($now - $ipTime) -gt 300000) {
                try {
                    $r = Invoke-RestMethod -Uri 'http://ip-api.com/json/?fields=status,lat,lon' -TimeoutSec 3
                    if ($r.status -eq 'success') {
                        $ipCoord = "$( ([double]$r.lat).ToString('R', $culture) )|$( ([double]$r.lon).ToString('R', $culture) )"
                        $ipTime = $now
                    }
                } catch {}
            }
            if ($null -ne $ipCoord) {
                [Console]::WriteLine("$ipCoord|$($now.ToString($culture))")
            } else {
                [Console]::WriteLine('unavailable')
            }
        } else {
            [Console]::WriteLine('unavailable')
        }
        Start-Sleep -Seconds 1
    }
} finally {
    $watcher.Dispose()
}
"""


class DeviceLocation:
    def __init__(self, max_age=60):
        self.max_age = max_age
        self._fix = None
        self._process = None
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        if os.name != "nt":
            print("Device location requires Windows; photos will use location-unavailable.")
            return
        try:
            self._process = subprocess.Popen(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                 "-Command", LOCATION_SCRIPT],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError as error:
            print(f"Device location unavailable: {error}")
            return
        self._thread = threading.Thread(target=self._read_updates, daemon=True)
        self._thread.start()
        print("Tracking device location. Enable Windows Settings > Privacy & security > Location "
              "if photos show location-unavailable.")

    def _read_updates(self):
        try:
            for line in self._process.stdout:
                self._update(line)
        finally:
            with self._lock:
                self._fix = None

    def _update(self, line):
        try:
            lat, lon, timestamp_ms = map(float, line.strip().split("|"))
            if not (math.isfinite(timestamp_ms) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Invalid location")
            fix = (lat, lon, timestamp_ms / 1000)
        except ValueError:
            fix = None
        with self._lock:
            self._fix = fix

    def label(self):
        with self._lock:
            fix = self._fix
        if fix is None or not 0 <= time.time() - fix[2] <= self.max_age:
            return "location-unavailable"
        return f"{fix[0]:.6f}_{fix[1]:.6f}"

    def stop(self):
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            self._process.wait()
            self._thread.join(timeout=2)
            self._process.stdout.close()
            self._process = None
        with self._lock:
            self._fix = None


def photo_filename(location_label, captured_at=None):
    """Include local time, microseconds and UTC offset without Windows-invalid colons."""
    if captured_at is None:
        captured_at = datetime.now().astimezone()
    return f"[{location_label}], [{captured_at:%Y-%m-%d_%H-%M-%S.%f%z}].jpg"
