/** AI-TextAnalyze frontend — pywebview bridge client */

let currentFilter = "all";
let showRaw = false;
let allKeywords = [];
let modelStatus = { ner_available: false, mt_available: false };

const TYPE_LABELS = {
  per: "人名",
  country: "国名",
  city: "地名",
  org: "組織",
};

function hasApi() {
  return typeof pywebview !== "undefined" && pywebview.api;
}

async function apiCall(method, ...args) {
  if (!hasApi()) {
    return { ok: false, error: "バックエンド API に接続できません（pywebview 外で開いています）" };
  }
  try {
    return await pywebview.api[method](...args);
  } catch (err) {
    return { ok: false, error: String(err) };
  }
}

function setTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  document.querySelectorAll("#themeToggle button").forEach((b) => {
    b.classList.toggle("active", b.dataset.t === t);
  });
}

function status(msg) {
  document.getElementById("statusMsg").textContent = msg;
}

function showProgress(message, current = 0, total = 0) {
  const overlay = document.getElementById("progressOverlay");
  const label = document.getElementById("progressLabel");
  const bar = document.getElementById("progressBar");
  const detail = document.getElementById("progressDetail");
  overlay.style.display = "flex";
  label.textContent = message || "処理中…";
  bar.classList.remove("indeterminate");
  if (total > 0) {
    const pct = Math.min(100, Math.round((current / total) * 100));
    bar.style.width = pct + "%";
    detail.textContent = `${current} / ${total}`;
  } else {
    bar.style.width = "35%";
    bar.classList.add("indeterminate");
    detail.textContent = "";
  }
}

function hideProgress() {
  const overlay = document.getElementById("progressOverlay");
  const bar = document.getElementById("progressBar");
  overlay.style.display = "none";
  bar.classList.remove("indeterminate");
  bar.style.width = "0%";
  document.getElementById("progressDetail").textContent = "";
}

function updateModelBar() {
  const ner = modelStatus.ner_available
    ? `NER: ${modelStatus.ner_name || "検出"}`
    : "NER: 未検出";
  const mt = modelStatus.mt_available
    ? `翻訳: ${modelStatus.mt_name || "検出"}`
    : "翻訳: 未検出";
  document.getElementById("modelStatus").innerHTML =
    `<span class="status-dot${modelStatus.ner_available && modelStatus.mt_available ? "" : " warn"}" id="statusDot"></span>${ner}  ·  ${mt}`;

  document.getElementById("btnExtract").disabled = !modelStatus.ner_available;
  document.getElementById("btnTranslate").disabled = !modelStatus.mt_available;
}

async function initApp() {
  const res = await apiCall("get_model_status");
  if (res.ok) {
    modelStatus = res;
    updateModelBar();
    status("準備完了");
  } else {
    status(res.error || "モデル状態の取得に失敗");
  }
}

function updateCount() {
  const n = document.getElementById("source").value.length;
  document.getElementById("charCount").textContent = n.toLocaleString() + " 文字";
}

async function onSourceInput() {
  updateCount();
  const text = document.getElementById("source").value;
  if (!text.trim()) {
    document.getElementById("detLang").textContent = "—";
    return;
  }
  const res = await apiCall("detect_language", text);
  if (res.ok) {
    document.getElementById("detLang").textContent = res.display || res.language;
  }
}

async function attachFile() {
  status("ファイルを選択…");
  const res = await apiCall("pick_and_read_file");
  if (!res.ok) {
    status(res.error || "読み込み失敗");
    return;
  }
  if (res.cancelled) {
    status("キャンセルしました");
    return;
  }
  document.getElementById("source").value = res.text || "";
  updateCount();
  if (res.display) {
    document.getElementById("detLang").textContent = res.display;
  } else if (res.language) {
    document.getElementById("detLang").textContent = res.language;
  }
  status(`読み込み完了: ${res.path || ""}`);
}

function clearAll() {
  document.getElementById("source").value = "";
  document.getElementById("target").value = "";
  allKeywords = [];
  document.getElementById("kwList").innerHTML =
    '<div class="kw-empty" style="padding:24px 14px;color:var(--text-sub);font-size:12.5px;text-align:center;">「キーワード抽出」を実行すると<br>ここに人名・国名・地名・組織が並びます。</div>';
  document.getElementById("kwRaw").textContent = "";
  document.getElementById("kwCount").textContent = "0 件";
  document.getElementById("kwHint").textContent = "—";
  updateCount();
  document.getElementById("detLang").textContent = "—";
  status("クリアしました");
}

async function extract() {
  const text = document.getElementById("source").value.trim();
  if (!text) {
    status("本文が空です");
    return;
  }
  showProgress("キーワード抽出中…", 0, 0);
  status("抽出中…");
  const res = await apiCall("extract_keywords", text);
  hideProgress();
  if (!res.ok) {
    status(res.error || "抽出失敗");
    return;
  }
  allKeywords = res.keywords || [];
  renderKw();
  status(`抽出完了（${allKeywords.length} 件）`);
}

function renderKw() {
  const list = allKeywords.filter(
    (k) => currentFilter === "all" || k.type === currentFilter
  );
  const box = document.getElementById("kwList");
  if (!list.length) {
    box.innerHTML =
      '<div style="padding:24px 14px;color:var(--text-sub);font-size:12.5px;text-align:center;">該当キーワードがありません</div>';
  } else {
    box.innerHTML = list
      .map(
        (k) => `
      <div class="kw-item">
        <span class="kw-badge b-${k.type}">${TYPE_LABELS[k.type] || k.type}</span>
        <span class="kw-term">${escapeHtml(k.term)}</span>
        <span class="kw-freq">×${k.freq}</span>
      </div>`
      )
      .join("");
  }
  document.getElementById("kwCount").textContent = list.length + " 件";
  document.getElementById("kwHint").textContent = "全 " + allKeywords.length + " 件抽出";
  document.getElementById("kwRaw").textContent = list.map((k) => k.term).join("\n");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function filterKw(btn) {
  document.querySelectorAll(".chip[data-f]").forEach((c) => c.classList.remove("on"));
  btn.classList.add("on");
  currentFilter = btn.dataset.f;
  renderKw();
}

function toggleRaw() {
  showRaw = !showRaw;
  document.getElementById("kwList").style.display = showRaw ? "none" : "block";
  document.getElementById("kwRaw").style.display = showRaw ? "block" : "none";
}

function copyRaw() {
  const list = allKeywords.filter(
    (k) => currentFilter === "all" || k.type === currentFilter
  );
  const txt = list.map((k) => k.term).join("\n");
  navigator.clipboard?.writeText(txt);
  status("リストをコピーしました");
}

async function translateText() {
  const src = document.getElementById("source").value.trim();
  if (!src) {
    status("本文が空です");
    return;
  }
  const srcLang = document.getElementById("srcLang").value;
  const tgtLang = document.getElementById("tgtLang").value;
  showProgress("翻訳中…", 0, 0);
  status("翻訳中…");
  const res = await apiCall("translate", src, srcLang, tgtLang);
  hideProgress();
  if (!res.ok) {
    status(res.error || "翻訳失敗");
    return;
  }
  document.getElementById("target").value = res.text || "";
  status("翻訳完了");
}

function swapLang() {
  const s = document.getElementById("srcLang");
  const t = document.getElementById("tgtLang");
  if (s.value === "auto") {
    status("自動判定は入れ替えできません");
    return;
  }
  [s.value, t.value] = [t.value, s.value];
  status("言語を入れ替えました");
}

window.addEventListener("pywebviewready", initApp);
if (hasApi()) {
  initApp();
}

// pywebview bridge から呼ばれる
window.status = status;
window.showProgress = showProgress;
window.hideProgress = hideProgress;
