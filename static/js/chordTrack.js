import { onLanguageChange, t } from "./i18n.js";

const CHORD_SYMBOL_RE = /^(?:N\.C\.|[A-G](?:#|b)?(?:m7|maj7|sus4|dim|aug|m|7)?(?:\/[A-G](?:#|b)?)?)$/;
const SAVE_DELAY_MS = 600;

let trackId = null;
let duration = 0;
let chords = [];
let ribbon = null;
let container = null;
let saveIndicator = null;
let saveButton = null;
let saveTimer = null;
let savedTimer = null;
let saveChain = Promise.resolve();
let loadToken = 0;
let bridge = null;

export function normalizeChordEvents(raw, songDuration) {
  const limit = Number(songDuration);
  if (!Array.isArray(raw) || !Number.isFinite(limit) || limit <= 0) return [];

  const ordered = raw
    .filter((item) => item && typeof item === "object")
    .map((item) => ({
      start: Number(item.start),
      end: Math.min(Number(item.end), limit),
      symbol: typeof item.symbol === "string" ? item.symbol.trim() : "",
      confidence: Number.isFinite(Number(item.confidence))
        ? Math.max(0, Math.min(1, Number(item.confidence)))
        : 1,
    }))
    .filter((item) => (
      Number.isFinite(item.start)
      && Number.isFinite(item.end)
      && item.start >= 0
      && item.start < limit
      && item.end > item.start
      && CHORD_SYMBOL_RE.test(item.symbol)
    ))
    .sort((a, b) => a.start - b.start);

  const clean = [];
  for (const event of ordered) {
    const previous = clean.at(-1);
    if (previous && event.start < previous.end) {
      event.start = previous.end;
      if (event.end <= event.start) continue;
    }
    clean.push(event);
  }
  return clean;
}

function indicator() {
  return saveIndicator || document.getElementById("chordTrackSaveIndicator");
}

function showSaving() {
  const node = indicator();
  if (!node) return;
  clearTimeout(savedTimer);
  node.textContent = t("chords.saving");
  node.className = "sections-save-indicator";
}

function showSaved() {
  const node = indicator();
  if (!node) return;
  node.textContent = t("chords.saved");
  node.className = "sections-save-indicator saved";
  savedTimer = setTimeout(() => {
    node.className = "sections-save-indicator hidden";
  }, 1800);
}

function showSaveFailed() {
  const node = indicator();
  if (!node) return;
  clearTimeout(savedTimer);
  node.textContent = t("chords.saveFailed");
  node.className = "sections-save-indicator notice";
}

function clearSaveIndicator() {
  const node = indicator();
  if (node) node.className = "sections-save-indicator hidden";
  clearTimeout(savedTimer);
}

function body() {
  return JSON.stringify({ chords });
}

function queueSave() {
  if (!trackId) return saveChain;
  const id = trackId;
  const snapshot = body();
  saveChain = saveChain.then(() => sendSave(id, snapshot));
  return saveChain;
}

async function sendSave(id, snapshot) {
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(id)}/lead-sheet/chords`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: snapshot,
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => String(response.status));
      console.warn("[chords] save failed:", response.status, detail);
      if (id === trackId) showSaveFailed();
      return;
    }
    if (id === trackId && snapshot === body() && saveTimer === null) showSaved();
  } catch (error) {
    console.warn("[chords] save failed:", error);
    if (id === trackId) showSaveFailed();
  }
}

function scheduleSave() {
  clearTimeout(saveTimer);
  showSaving();
  saveTimer = setTimeout(() => {
    saveTimer = null;
    queueSave();
  }, SAVE_DELAY_MS);
}

function applyState(state) {
  const metadata = state?.metadata;
  duration = Number(metadata?.duration) || 0;
  chords = normalizeChordEvents(metadata?.chords, duration);
  if (ribbon) ribbon.classList.toggle("hidden", !state?.has_lead_sheet);
  if (saveButton) {
    saveButton.disabled = !state?.has_lead_sheet || duration <= 0 || state.status === "running";
  }
  render();
}

function render() {
  if (!container) return;
  container.replaceChildren();
  if (!duration) return;

  chords.forEach((chord, index) => {
    const mark = document.createElement("div");
    mark.className = "chord-event";
    mark.dataset.index = String(index);
    mark.style.left = `${(chord.start / duration * 100).toFixed(4)}%`;
    mark.style.width = `${((chord.end - chord.start) / duration * 100).toFixed(4)}%`;
    mark.title = `${chord.symbol} · ${formatTime(chord.start)}`;

    const label = document.createElement("button");
    label.type = "button";
    label.className = "chord-label";
    label.textContent = chord.symbol;
    label.title = t("chords.editTitle", { symbol: chord.symbol });
    label.setAttribute("aria-label", t("chords.editTitle", { symbol: chord.symbol }));
    label.addEventListener("click", () => bridge?.seek?.(chord.start));
    label.addEventListener("dblclick", (event) => {
      event.preventDefault();
      event.stopPropagation();
      openEditor(index);
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chord-delete";
    remove.textContent = "×";
    remove.title = t("chords.deleteTitle", { symbol: chord.symbol });
    remove.setAttribute("aria-label", remove.title);
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      chords.splice(index, 1);
      render();
      scheduleSave();
    });

    mark.append(label, remove);
    container.append(mark);
  });
}

function formatTime(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function openEditor(index, rollback = null) {
  const mark = container?.querySelector(`.chord-event[data-index="${index}"]`);
  const label = mark?.querySelector(".chord-label");
  const chord = chords[index];
  if (!label || !chord) return;

  const before = rollback || chords.map((item) => ({ ...item }));
  const input = document.createElement("input");
  input.className = "chord-edit-input";
  input.type = "text";
  input.maxLength = 32;
  input.value = chord.symbol;
  input.setAttribute("aria-label", t("chords.editTitle", { symbol: chord.symbol }));
  label.replaceWith(input);
  input.focus();
  input.select();

  let closed = false;
  const cancel = () => {
    if (closed) return;
    closed = true;
    chords = before;
    render();
  };
  const commit = () => {
    if (closed) return;
    const symbol = input.value.trim();
    if (!CHORD_SYMBOL_RE.test(symbol)) {
      input.setCustomValidity(t("chords.invalidSymbol"));
      input.reportValidity();
      input.focus();
      return;
    }
    closed = true;
    chords[index] = { ...chords[index], symbol };
    render();
    if (body() !== JSON.stringify({ chords: before })) scheduleSave();
  };
  input.addEventListener("blur", commit);
  input.addEventListener("input", () => input.setCustomValidity(""));
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      input.blur();
    } else if (event.key === "Escape") {
      event.preventDefault();
      input.removeEventListener("blur", commit);
      cancel();
    }
  });
}

function addChordAtPlayhead() {
  const time = Number(bridge?.getTime?.());
  if (!Number.isFinite(time) || duration <= 0 || time >= duration) return;

  const at = Math.max(0, time);
  const before = chords.map((chord) => ({ ...chord }));
  let index = chords.findIndex((chord) => chord.start <= at && at < chord.end);
  if (index >= 0) {
    const current = chords[index];
    if (at - current.start > 0.05) {
      chords[index] = { ...current, end: at };
      chords.splice(index + 1, 0, { ...current, start: at, confidence: 1 });
      index++;
    }
  } else {
    const next = chords.find((chord) => chord.start > at);
    const end = next?.start ?? duration;
    if (end <= at) return;
    index = chords.findIndex((chord) => chord.start > at);
    if (index < 0) index = chords.length;
    chords.splice(index, 0, { start: at, end, symbol: "C", confidence: 1 });
  }

  render();
  openEditor(index, before);
}

async function load(id) {
  clearTimeout(saveTimer);
  if (trackId && saveTimer !== null) {
    saveTimer = null;
    queueSave();
  }
  const token = ++loadToken;
  trackId = id || null;
  duration = 0;
  chords = [];
  clearSaveIndicator();
  if (saveButton) saveButton.disabled = true;
  render();
  if (ribbon) ribbon.classList.add("hidden");
  if (!trackId) return;

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(trackId)}/lead-sheet`, {
      cache: "no-store",
    });
    if (response.status === 404 || response.status === 409) return;
    if (!response.ok) throw new Error(`lead sheet -> ${response.status}`);
    const state = await response.json();
    if (token === loadToken && trackId === id) applyState(state);
  } catch (error) {
    if (token !== loadToken) return;
    console.warn("[chords] lead sheet load failed:", error);
  }
}

export function setChordTrackBridge(value) {
  bridge = value;
}

export function initChordTrack() {
  ribbon = document.getElementById("daw-chords-ribbon");
  container = document.getElementById("daw-chords-track");
  saveIndicator = document.getElementById("chordTrackSaveIndicator");
  saveButton = document.getElementById("chordTrackAdd");
  if (!ribbon || !container || !saveButton || !saveIndicator) return;

  saveButton.addEventListener("click", addChordAtPlayhead);
  document.addEventListener("trackopen", (event) => load(event.detail?.id));
  document.addEventListener("leadsheetchange", (event) => {
    if (event.detail?.trackId !== trackId) return;
    if (event.detail.state?.status === "running" && event.detail.state?.has_lead_sheet) {
      if (saveButton) saveButton.disabled = true;
      return;
    }
    applyState(event.detail.state);
  });
  onLanguageChange(render);
}
