/**
 * Windows-exclusive download CTA for Dana GitHub Releases.
 */
(function () {
  "use strict";

  var DOWNLOAD_HREF =
    "https://github.com/Cascade-Router/Dana/releases/latest/download/Dana-windows-x64.zip";
  var DOWNLOAD_LABEL = "Download for Windows (RTX Required)";

  var ASSETS = {
    windows: {
      label: DOWNLOAD_LABEL,
      href: DOWNLOAD_HREF,
    },
  };

  function applyDownloadButton() {
    var btn = document.getElementById("download-btn");
    if (!btn) {
      return;
    }

    btn.textContent = ASSETS.windows.label;
    btn.setAttribute("href", ASSETS.windows.href);
    btn.dataset.os = "windows";

    var hint = document.getElementById("os-hint");
    if (hint) {
      hint.textContent =
        "Windows 10/11 · NVIDIA RTX GPU required (8GB+ VRAM)";
    }
  }

  // Expose for manual verification / tests.
  window.DanaLanding = {
    applyDownloadButton: applyDownloadButton,
    ASSETS: ASSETS,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyDownloadButton);
  } else {
    applyDownloadButton();
  }
})();
