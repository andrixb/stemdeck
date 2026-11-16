// The Lead Sheet tab: choose one or more of the open track's pitched stems
// and transcribe each one as a separate part, with shared chord symbols.
//
// The server (app/pipeline/lead_sheet.py) transcribes selected stems and
// writes MusicXML/MIDI beside the song. This module starts a run, polls it,
// and shows key, tempo, counts, chord symbols, and the first notes. The files
// themselves are downloads, opened in a notation app rather than in the page.
//
// Nothing is fetched from the network and no lyrics or tags are read: the
// lead sheet is made from the audio StemDeck already separated.

import { t } from "./i18n.js";
import { getCurrentTrackInfo } from "./catalog.js";

let visible = false;
let shownTrackId = null;
let pollTimer = 0;
let loadToken = 0;
let statusEl = null;
let actionsEl = null;
let bodyEl = null;
let dialogEl = null;
let stemOptionsEl = null;
let dialogErrorEl = null;
let pendingSelection = null;

const POLL_MS = 1200;
// Enough of a long transcription to see the shape without turning the
// sidebar into a score viewer. The files carry everything.
const MAX_CHORDS = 64;
const MAX_NOTES = 160;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function setStatus(message = "", kind = "") {
  statusEl.textContent = message;
  statusEl.className = `leadsheet-status${kind ? ` ${kind}` : ""}`;
}

function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = 0;
}

function schedulePoll(trackId) {
  stopPolling();
  pollTimer = setTimeout(() => load(trackId), POLL_MS);
}

function formatTime(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function apiBase(trackId) {
  return `/api/jobs/${encodeURIComponent(trackId)}/lead-sheet`;
}

async function fetchState(trackId) {
  const res = await fetch(apiBase(trackId), { cache: "no-store" });
  if (res.status === 404 || res.status === 409) return null;
  if (!res.ok) throw new Error(`lead sheet -> ${res.status}`);
  return res.json();
}

function downloadLink(label, url, name) {
  const link = el("a", "leadsheet-btn", label);
  link.href = url;
  link.download = name;
  return link;
}

function actionButton(label, onClick, { primary = false, disabled = false } = {}) {
  const button = el("button", `leadsheet-btn${primary ? " primary" : ""}`, label);
  button.type = "button";
  button.disabled = disabled;
  button.addEventListener("click", onClick);
  return button;
}

function openSelectionDialog(trackId, force, state) {
  const available = state.available_melody_stems || [];
  if (!available.length) {
    setStatus(t("leadSheet.noMelodyStems"), "error");
    return;
  }
  const selected = new Set(
    (state.metadata?.melody_stems || []).filter((name) => available.includes(name)),
  );
  stemOptionsEl.replaceChildren();
  for (const stem of available) {
    const label = el("label", "leadsheet-stem-option");
    const checkbox = el("input");
    checkbox.type = "checkbox";
    checkbox.name = "melodyStem";
    checkbox.value = stem;
    checkbox.checked = selected.has(stem);
    checkbox.addEventListener("change", () => { dialogErrorEl.textContent = ""; });
    label.append(checkbox, el("span", null, t(`leadSheet.stem.${stem}`)));
    stemOptionsEl.append(label);
  }
  dialogErrorEl.textContent = "";
  pendingSelection = { trackId, force };
  dialogEl.showModal();
}

function renderActions(state, trackId) {
  actionsEl.replaceChildren();
  if (state.status === "running") {
    actionsEl.append(actionButton(t("leadSheet.working"), () => {}, { disabled: true }));
    return;
  }
  const files = state.files || {};
  if (files["lead-sheet.musicxml"]) {
    actionsEl.append(
      downloadLink(t("leadSheet.downloadMusicxml"), files["lead-sheet.musicxml"], "lead-sheet.musicxml"),
      downloadLink(t("leadSheet.downloadMidi"), files["lead-sheet.mid"], "lead-sheet.mid"),
    );
  }
  const has = Boolean(state.has_lead_sheet);
  actionsEl.append(
    actionButton(
      has ? t("leadSheet.regenerate") : t("leadSheet.generate"),
      () => openSelectionDialog(trackId, has, state),
      { primary: !has },
    ),
  );
  if (has) {
    actionsEl.append(actionButton(t("leadSheet.delete"), () => remove(trackId)));
  }
}

function fact(label, value) {
  const wrap = el("div", "leadsheet-fact");
  wrap.append(el("span", "leadsheet-fact-label", label), el("span", "leadsheet-fact-value", value ?? "—"));
  return wrap;
}

function renderBody(state) {
  bodyEl.replaceChildren();
  const meta = state.metadata;
  if (!meta) return;
  const counts = meta.counts || {};

  const summary = el("div", "leadsheet-summary");
  summary.append(
    fact(t("leadSheet.key"), meta.key?.name),
    fact(t("leadSheet.tempo"), meta.tempo ? `${Math.round(meta.tempo)} BPM` : null),
    fact(t("leadSheet.meter"), meta.time_signature),
    fact(t("leadSheet.bars"), counts.measures),
    fact(t("leadSheet.notes"), counts.notes),
    fact(t("leadSheet.chords"), counts.chords),
  );
  bodyEl.append(summary);

  const chords = (meta.chords || []).filter((chord) => chord.symbol && chord.symbol !== "N.C.");
  if (chords.length) {
    bodyEl.append(el("p", "leadsheet-section-label", t("leadSheet.chordSymbols")));
    const row = el("div", "leadsheet-chords");
    for (const chord of chords.slice(0, MAX_CHORDS)) {
      const chip = el("span", "leadsheet-chip");
      chip.append(chord.symbol, el("small", null, ` ${formatTime(chord.start)}`));
      row.append(chip);
    }
    bodyEl.append(row);
  }

  const notes = meta.notes || [];
  if (notes.length) {
    bodyEl.append(el("p", "leadsheet-section-label", t("leadSheet.melody")));
    const list = el("div", "leadsheet-notes");
    for (const note of notes.slice(0, MAX_NOTES)) {
      list.append(el("span", "at", formatTime(note.start)), el("span", null, note.name));
    }
    bodyEl.append(list);
    if (notes.length > MAX_NOTES) {
      bodyEl.append(el("p", "leadsheet-path-hint", t("leadSheet.moreNotes", { count: notes.length - MAX_NOTES })));
    }
  }

  for (const warning of meta.warnings || []) {
    bodyEl.append(el("p", "leadsheet-warning", warning));
  }

  bodyEl.append(el("p", "leadsheet-path-hint", t("leadSheet.openHint")));
}

function render(state, trackId) {
  if (!state) {
    actionsEl.replaceChildren();
    bodyEl.replaceChildren();
    setStatus(t("leadSheet.unavailable"), "muted");
    return;
  }
  if (state.status === "running") {
    setStatus(t("leadSheet.running"), "muted");
  } else if (state.status === "error") {
    setStatus(state.error || t("leadSheet.failed"), "error");
  } else if (state.has_lead_sheet) {
    setStatus(t("leadSheet.ready"), "muted");
  } else {
    setStatus(t("leadSheet.empty"), "muted");
  }
  renderActions(state, trackId);
  renderBody(state);
}

async function load(trackId = getCurrentTrackInfo()?.id) {
  if (!visible) return;
  stopPolling();
  const info = getCurrentTrackInfo();
  const id = trackId || info?.id;
  if (!id) {
    shownTrackId = null;
    actionsEl.replaceChildren();
    bodyEl.replaceChildren();
    setStatus(t("leadSheet.openTrack"), "muted");
    return;
  }
  if (id !== shownTrackId) {
    shownTrackId = id;
    actionsEl.replaceChildren();
    bodyEl.replaceChildren();
    setStatus(t("leadSheet.loading"), "muted");
  }
  const token = ++loadToken;
  let state = null;
  try {
    state = await fetchState(id);
  } catch (err) {
    if (token !== loadToken) return;
    console.warn("lead sheet state failed", err);
    setStatus(t("leadSheet.offline"), "error");
    return;
  }
  if (token !== loadToken || !visible || getCurrentTrackInfo()?.id !== id) return;
  render(state, id);
  if (state?.status === "running") schedulePoll(id);
}

async function generate(trackId, force, melodyStems) {
  if (!visible || !trackId) return;
  setStatus(t("leadSheet.starting"), "muted");
  try {
    const res = await fetch(apiBase(trackId), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: Boolean(force), melody_stems: melodyStems }),
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      setStatus(data?.detail || t("leadSheet.failed"), "error");
      return;
    }
    if (getCurrentTrackInfo()?.id !== trackId) return;
    shownTrackId = trackId;
    render(data, trackId);
    if (data?.status === "running") schedulePoll(trackId);
  } catch (err) {
    console.warn("lead sheet start failed", err);
    setStatus(t("leadSheet.offline"), "error");
  }
}

async function remove(trackId) {
  if (!visible || !trackId) return;
  try {
    const res = await fetch(apiBase(trackId), { method: "DELETE" });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      setStatus(data?.detail || t("leadSheet.failed"), "error");
      return;
    }
    if (getCurrentTrackInfo()?.id !== trackId) return;
    render(data, trackId);
    if (data?.status === "running") schedulePoll(trackId);
  } catch (err) {
    console.warn("lead sheet delete failed", err);
    setStatus(t("leadSheet.offline"), "error");
  }
}

function setVisible(on) {
  visible = on;
  if (!on) {
    if (dialogEl?.open) dialogEl.close();
    loadToken++;
    stopPolling();
    return;
  }
  load();
}

export function initLeadSheet() {
  statusEl = document.getElementById("leadSheetStatus");
  actionsEl = document.getElementById("leadSheetActions");
  bodyEl = document.getElementById("leadSheetBody");
  dialogEl = document.getElementById("leadSheetDialog");
  stemOptionsEl = document.getElementById("leadSheetStemOptions");
  dialogErrorEl = document.getElementById("leadSheetDialogError");
  const cancelButton = document.getElementById("leadSheetCancel");
  const confirmButton = document.getElementById("leadSheetConfirm");
  if (
    !statusEl || !actionsEl || !bodyEl || !dialogEl || !stemOptionsEl ||
    !dialogErrorEl || !cancelButton || !confirmButton
  ) return;

  cancelButton.addEventListener("click", () => dialogEl.close());
  confirmButton.addEventListener("click", () => {
    const melodyStems = Array.from(
      stemOptionsEl.querySelectorAll('input[name="melodyStem"]:checked'),
      (checkbox) => checkbox.value,
    );
    if (!melodyStems.length) {
      dialogErrorEl.textContent = t("leadSheet.selectAtLeastOne");
      return;
    }
    const request = pendingSelection;
    dialogEl.close();
    if (request) generate(request.trackId, request.force, melodyStems);
  });
  dialogEl.addEventListener("close", () => {
    pendingSelection = null;
    dialogErrorEl.textContent = "";
  });

  document.addEventListener("catalogviewchange", (e) => setVisible(e.detail?.view === "leadsheet"));
  // A song opened while the tab is showing replaces what is on screen.
  document.addEventListener("trackopen", () => {
    if (visible) {
      if (dialogEl.open) dialogEl.close();
      load();
    }
  });
}
