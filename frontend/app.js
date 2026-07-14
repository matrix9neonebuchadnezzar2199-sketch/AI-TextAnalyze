/** AI-TextAnalyze frontend — pywebview bridge client */

let currentFilter = "all";
let showRaw = false;
let allKeywords = [];
let selectedKeywordKeys = new Set();
let alignmentUnits = [];
let activeUnitId = null;
let sourceHighlightDebounce = null;
/** 翻訳後は本文も unit ブロック表示（正規化差分で indexOf が外れるのを避ける） */
let sourceAlignMode = false;
let sourcePlainBackup = "";
let modelStatus = {
  ner_available: false,
  mt_available: false,
  mt_models: [],
  selected_mt_id: "nllb-600m",
};

const MT_STORAGE_KEY = "ai-textanalyze-mt-model";
const THEME_STORAGE_KEY = "ai-textanalyze-theme";
const SOURCE_SIDEBAR_KEY = "ai-textanalyze-source-collapsed";
const SOURCE_WIDTH_KEY = "ai-textanalyze-source-width";

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

function keywordKey(kw) {
  return `${kw.type}|${kw.term.toLowerCase()}`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeRegex(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getSourceEl() {
  return document.getElementById("source");
}

function getSourceText() {
  if (sourceAlignMode) return sourcePlainBackup;
  return getSourceEl().innerText || "";
}

function setSourceText(text) {
  exitSourceAlignMode();
  const el = getSourceEl();
  el.textContent = text || "";
  refreshSourceDisplay();
}

function exitSourceAlignMode() {
  if (!sourceAlignMode) return;
  sourceAlignMode = false;
  const view = document.getElementById("sourceAlignView");
  if (view) {
    view.hidden = true;
    view.innerHTML = "";
  }
  const el = getSourceEl();
  el.hidden = false;
  el.contentEditable = "true";
}

function enterSourceAlignMode(units) {
  sourcePlainBackup = getSourceEl().innerText || sourcePlainBackup;
  sourceAlignMode = true;
  const el = getSourceEl();
  el.hidden = true;
  el.contentEditable = "false";
  const view = document.getElementById("sourceAlignView");
  if (view) view.hidden = false;
  renderSourceUnits(units);
}

function getTargetPlainText() {
  if (!alignmentUnits.length) {
    const view = document.getElementById("targetView");
    return view?.innerText?.trim() ? view.innerText : "";
  }
  return alignmentUnits.map((u) => u.tgt).join("\n\n");
}

function setTheme(t) {
  const theme = t === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", theme);
  document.querySelectorAll("#themeToggle button").forEach((b) => {
    b.classList.toggle("active", b.dataset.t === theme);
  });
  localStorage.setItem(THEME_STORAGE_KEY, theme);
}

function applySourceCollapsed(collapsed) {
  const workspace = document.getElementById("workspace");
  const btn = document.getElementById("btnToggleSource");
  if (!workspace) return;
  workspace.classList.toggle("source-collapsed", collapsed);
  if (btn) {
    btn.title = collapsed ? "本文を展開" : "本文を折りたたむ";
    btn.setAttribute("aria-label", collapsed ? "本文サイドバーを展開" : "本文サイドバーを折りたたむ");
    btn.textContent = collapsed ? "▶" : "◀";
  }
  localStorage.setItem(SOURCE_SIDEBAR_KEY, collapsed ? "1" : "0");
}

function toggleSourceSidebar() {
  const workspace = document.getElementById("workspace");
  if (!workspace) return;
  applySourceCollapsed(!workspace.classList.contains("source-collapsed"));
}

function initSplitter() {
  const splitter = document.getElementById("splitter");
  const workspace = document.getElementById("workspace");
  if (!splitter || !workspace) return;

  const saved = localStorage.getItem(SOURCE_WIDTH_KEY);
  if (saved) {
    const px = parseInt(saved, 10);
    if (px >= 220 && px <= 900) {
      workspace.style.setProperty("--source-col-width", `${px}px`);
    }
  }

  let dragging = false;

  splitter.addEventListener("mousedown", (e) => {
    if (workspace.classList.contains("source-collapsed")) return;
    dragging = true;
    splitter.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = workspace.getBoundingClientRect();
    const width = Math.min(900, Math.max(220, e.clientX - rect.left - 12));
    workspace.style.setProperty("--source-col-width", `${width}px`);
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    const val = workspace.style.getPropertyValue("--source-col-width");
    if (val) {
      const px = parseInt(val, 10);
      if (!Number.isNaN(px)) localStorage.setItem(SOURCE_WIDTH_KEY, String(px));
    }
  });
}

function status(msg) {
  document.getElementById("statusMsg").textContent = msg;
}

function showProgress(message, current = 0, total = 0, detail = "") {
  const overlay = document.getElementById("progressOverlay");
  const label = document.getElementById("progressLabel");
  const bar = document.getElementById("progressBar");
  const detailEl = document.getElementById("progressDetail");
  overlay.style.display = "flex";
  label.textContent = message || "処理中…";
  bar.classList.remove("indeterminate");
  if (total > 0) {
    const pct = Math.min(100, Math.round((current / total) * 100));
    bar.style.width = pct + "%";
    detailEl.textContent = detail || `${current} / ${total}`;
  } else {
    bar.style.width = "35%";
    bar.classList.add("indeterminate");
    detailEl.textContent = detail || "";
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
  const mtLabel =
    (modelStatus.mt_models || []).find((m) => m.id === modelStatus.selected_mt_id)
      ?.label || modelStatus.mt_name || "未選択";
  const mt = modelStatus.mt_available ? `翻訳: ${mtLabel}` : `翻訳: ${mtLabel}（未インストール）`;
  document.getElementById("modelStatus").innerHTML =
    `<span class="status-dot${modelStatus.ner_available && modelStatus.mt_available ? "" : " warn"}" id="statusDot"></span>${ner}  ·  ${mt}`;

  document.getElementById("btnExtract").disabled = !modelStatus.ner_available;
  document.getElementById("btnTranslate").disabled = !modelStatus.mt_available;
}

function populateMtModelSelect() {
  const sel = document.getElementById("mtModelSelect");
  if (!sel || !modelStatus.mt_models?.length) return;

  const preferred =
    localStorage.getItem(MT_STORAGE_KEY) || modelStatus.selected_mt_id || "nllb-600m";
  sel.innerHTML = modelStatus.mt_models
    .map((m) => {
      const suffix = m.available ? "" : "（未インストール）";
      return `<option value="${m.id}"${m.available ? "" : " disabled"}>${m.label}${suffix}</option>`;
    })
    .join("");

  const canUsePreferred = modelStatus.mt_models.some(
    (m) => m.id === preferred && m.available
  );
  const fallback = modelStatus.mt_models.find((m) => m.available)?.id || preferred;
  sel.value = canUsePreferred ? preferred : fallback;
}

async function onMtModelChange() {
  const sel = document.getElementById("mtModelSelect");
  const modelId = sel.value;
  const previous = modelStatus.selected_mt_id || "nllb-600m";

  sel.disabled = true;
  document.getElementById("btnTranslate").disabled = true;

  const res = await apiCall("select_mt_model", modelId);
  hideProgress();
  sel.disabled = false;

  if (!res.ok) {
    sel.value = previous;
    status(res.error || "モデル切替失敗");
    updateModelBar();
    return;
  }

  modelStatus.selected_mt_id = res.selected_mt_id;
  modelStatus.mt_name = res.mt_name;
  modelStatus.mt_loaded = true;
  modelStatus.mt_available = true;
  localStorage.setItem(MT_STORAGE_KEY, modelId);
  updateModelBar();
  status(res.cached ? "翻訳モデル準備完了" : "翻訳モデル起動完了");
}

function initUiPrefs() {
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  setTheme(savedTheme === "light" ? "light" : "dark");
  applySourceCollapsed(localStorage.getItem(SOURCE_SIDEBAR_KEY) === "1");
  initSplitter();
}

function buildHighlightRanges(text, terms) {
  const ranges = [];
  for (const term of terms) {
    if (!term) continue;
    const re = new RegExp(escapeRegex(term), "gi");
    let match;
    while ((match = re.exec(text)) !== null) {
      ranges.push({ start: match.index, end: match.index + match[0].length, className: "kw-mark" });
      if (match.index === re.lastIndex) re.lastIndex++;
    }
  }
  ranges.sort((a, b) => a.start - b.start || b.end - a.end);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r.start < last.end) continue;
    merged.push(r);
  }
  return merged;
}

function applyRangesToHtml(text, ranges, extraClass = "") {
  if (!text) return "";
  if (!ranges.length) return escapeHtml(text);
  let html = "";
  let pos = 0;
  for (const r of ranges) {
    if (r.start > pos) html += escapeHtml(text.slice(pos, r.start));
    const cls = r.className + (extraClass ? ` ${extraClass}` : "");
    html += `<mark class="${cls}">${escapeHtml(text.slice(r.start, r.end))}</mark>`;
    pos = r.end;
  }
  if (pos < text.length) html += escapeHtml(text.slice(pos));
  return html;
}

function getSelectedTerms() {
  return allKeywords
    .filter((k) => selectedKeywordKeys.has(keywordKey(k)))
    .map((k) => k.term);
}

function refreshSourceDisplay() {
  if (sourceAlignMode) {
    renderSourceUnits(alignmentUnits);
    return;
  }
  const el = getSourceEl();
  const text = getSourceText();
  const terms = getSelectedTerms();
  const ranges = buildHighlightRanges(text, terms);
  const hadFocus = document.activeElement === el;
  el.innerHTML = applyRangesToHtml(text, ranges);
  if (hadFocus) el.focus();
}

function bindUnitClicks(root) {
  root.querySelectorAll(".mt-unit").forEach((node) => {
    node.addEventListener("click", () => onUnitClick(node.dataset.unitId));
    node.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onUnitClick(node.dataset.unitId);
      }
    });
  });
}

function renderUnitBlocks(view, units, field) {
  if (!view) return;
  if (!units?.length) {
    view.innerHTML =
      field === "tgt"
        ? '<div class="target-empty">翻訳結果がここに表示されます…</div>'
        : "";
    return;
  }
  const terms = getSelectedTerms();
  view.innerHTML = units
    .map((u) => {
      const text = u[field] || "";
      const active = u.id === activeUnitId ? " align-hl" : "";
      const ranges = buildHighlightRanges(text, terms);
      const inner = applyRangesToHtml(text, ranges);
      return `<div class="mt-unit${active}" data-unit-id="${escapeHtml(u.id)}" role="button" tabindex="0">${inner || "&nbsp;"}</div>`;
    })
    .join("");
  bindUnitClicks(view);
}

function renderTargetUnits(units) {
  renderUnitBlocks(document.getElementById("targetView"), units, "tgt");
}

function renderSourceUnits(units) {
  renderUnitBlocks(document.getElementById("sourceAlignView"), units, "src");
}

function scrollActiveUnitsIntoView() {
  if (!activeUnitId) return;
  document.querySelectorAll(".mt-unit.align-hl").forEach((node) => {
    node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
}

function onUnitClick(unitId) {
  if (activeUnitId === unitId) {
    activeUnitId = null;
  } else {
    activeUnitId = unitId;
  }
  renderTargetUnits(alignmentUnits);
  if (sourceAlignMode) {
    renderSourceUnits(alignmentUnits);
  } else {
    refreshSourceDisplay();
  }
  scrollActiveUnitsIntoView();
}

function refreshAllHighlights() {
  refreshSourceDisplay();
  if (alignmentUnits.length) renderTargetUnits(alignmentUnits);
}

async function initApp() {
  initUiPrefs();
  const sourceEl = getSourceEl();
  sourceEl.addEventListener("input", onSourceInput);
  sourceEl.addEventListener("paste", (e) => {
    e.preventDefault();
    const text = e.clipboardData?.getData("text/plain") || "";
    document.execCommand("insertText", false, text);
  });

  showProgress("起動中…", 0, 0, "モデルを確認しています");
  const res = await apiCall("get_model_status");
  if (res.ok) {
    modelStatus = res;
    populateMtModelSelect();
    updateModelBar();

    const sel = document.getElementById("mtModelSelect");
    const preferred = sel?.value || modelStatus.selected_mt_id;
    if (preferred && preferred !== modelStatus.selected_mt_id) {
      showProgress("モデル切替中…", 0, 0, preferred);
      await onMtModelChange();
      hideProgress();
      return;
    }

    if (res.mt_available && !res.mt_loaded) {
      showProgress("モデル読み込み中…", 0, 0, "翻訳エンジンを起動しています（初回のみ時間がかかります）");
      status("翻訳モデルを起動中…");
      const warm = await apiCall("warmup_mt", preferred);
      if (warm.ok) {
        modelStatus.mt_loaded = true;
        modelStatus.selected_mt_id = warm.selected_mt_id || preferred;
        status(warm.cached ? "翻訳モデル準備完了" : "翻訳モデル起動完了");
      } else if (warm.error) {
        status(warm.error);
      }
    } else {
      status("準備完了");
    }
  } else {
    status(res.error || "モデル状態の取得に失敗");
  }
  hideProgress();
}

function updateCount() {
  const n = getSourceText().length;
  document.getElementById("charCount").textContent = n.toLocaleString() + " 文字";
  const rail = document.getElementById("charCountRail");
  if (rail) rail.textContent = n.toLocaleString();
}

async function onSourceInput() {
  updateCount();
  if (sourceAlignMode) exitSourceAlignMode();
  alignmentUnits = [];
  activeUnitId = null;
  clearTimeout(sourceHighlightDebounce);
  sourceHighlightDebounce = setTimeout(() => refreshSourceDisplay(), 200);

  const text = getSourceText();
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
  alignmentUnits = [];
  activeUnitId = null;
  setSourceText(res.text || "");
  renderTargetUnits([]);
  updateCount();
  if (res.display) {
    document.getElementById("detLang").textContent = res.display;
  } else if (res.language) {
    document.getElementById("detLang").textContent = res.language;
  }
  status(`読み込み完了: ${res.path || ""}`);
}

function clearAll() {
  alignmentUnits = [];
  activeUnitId = null;
  setSourceText("");
  allKeywords = [];
  selectedKeywordKeys.clear();
  renderTargetUnits([]);
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
  const source = getSourceText().trim();
  const target = getTargetPlainText().trim();
  if (!source && !target) {
    status("本文・翻訳が空です");
    return;
  }
  status("抽出中…");
  const res = await apiCall("extract_keywords", source, target);
  hideProgress();
  if (!res.ok) {
    status(res.error || "抽出失敗");
    return;
  }
  allKeywords = res.keywords || [];
  selectedKeywordKeys.clear();
  renderKw();
  refreshAllHighlights();
  const scope = res.scope ? `（${res.scope}）` : "";
  status(`抽出完了（${allKeywords.length} 件${scope}）`);
}

function toggleKeywordSelection(kw) {
  const key = keywordKey(kw);
  if (selectedKeywordKeys.has(key)) {
    selectedKeywordKeys.delete(key);
  } else {
    selectedKeywordKeys.add(key);
  }
  renderKw();
  refreshAllHighlights();
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
      .map((k) => {
        const sel = selectedKeywordKeys.has(keywordKey(k)) ? " selected" : "";
        return `
      <div class="kw-item${sel}" data-key="${escapeHtml(keywordKey(k))}" role="button" tabindex="0">
        <span class="kw-badge b-${k.type}">${TYPE_LABELS[k.type] || k.type}</span>
        <span class="kw-term">${escapeHtml(k.term)}</span>
        <span class="kw-freq">×${k.freq}</span>
      </div>`;
      })
      .join("");

    box.querySelectorAll(".kw-item").forEach((node) => {
      const key = node.dataset.key;
      const kw = list.find((k) => keywordKey(k) === key);
      if (!kw) return;
      node.addEventListener("click", () => toggleKeywordSelection(kw));
      node.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggleKeywordSelection(kw);
        }
      });
    });
  }
  document.getElementById("kwCount").textContent = list.length + " 件";
  document.getElementById("kwHint").textContent = "全 " + allKeywords.length + " 件抽出";
  document.getElementById("kwRaw").textContent = list.map((k) => k.term).join("\n");
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

async function copyTextToClipboard(text, emptyMsg, okMsg) {
  const value = (text || "").trim();
  if (!value) {
    status(emptyMsg);
    return;
  }
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      const ta = document.createElement("textarea");
      ta.value = value;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    status(okMsg);
  } catch (err) {
    status("コピーに失敗しました: " + String(err));
  }
}

function copySource() {
  copyTextToClipboard(getSourceText(), "本文が空です", "本文をコピーしました");
}

function copyTarget() {
  copyTextToClipboard(getTargetPlainText(), "翻訳結果が空です", "翻訳をコピーしました");
}

async function translateText() {
  const src = getSourceText().trim();
  if (!src) {
    status("本文が空です");
    return;
  }
  const srcLang = document.getElementById("srcLang").value;
  const tgtLang = document.getElementById("tgtLang").value;
  status("翻訳中…");
  const res = await apiCall("translate", src, srcLang, tgtLang);
  hideProgress();
  if (!res.ok) {
    status(res.error || "翻訳失敗");
    return;
  }
  alignmentUnits = res.units || [];
  activeUnitId = null;
  // 本文を編集面に残したまま、unit ビューで src/tgt を ID 連動させる
  sourcePlainBackup = getSourceEl().innerText || src;
  enterSourceAlignMode(alignmentUnits);
  renderTargetUnits(alignmentUnits);
  updateCount();
  status("翻訳完了（クリックで本文⇔翻訳を対応表示）");
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

initUiPrefs();
window.addEventListener("pywebviewready", initApp);
if (hasApi()) {
  initApp();
} else {
  status("バックエンド未接続（プレビュー）");
  hideProgress();
}

window.status = status;
window.showProgress = showProgress;
window.hideProgress = hideProgress;
