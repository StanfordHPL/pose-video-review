const state = {
  trials: [],
  entries: [],
  poses: new Map(),
  offsets: new Map(),
  presentedFrames: new Map(),
  pendingFrames: new Map(),
  verification: new Map(),
  elapsed: 0,
  duration: 0,
  playing: false,
  animationFrame: null,
  lastSyncAt: 0,
  playbackStartedAt: 0,
  playbackStartElapsed: 0,
  loadRevision: 0,
};

const els = Object.fromEntries([
  "trialSelect", "showPose", "status", "cameraGrid", "playButton",
  "stepBackButton", "stepForwardButton", "timeSlider", "currentTime",
  "duration", "speedSelect", "loopPlayback", "tileSize", "tileSizeValue",
  "saveOffsetsButton", "resetOffsetsButton", "trialTypeFilter", "saveFilter",
  "markRoiStartButton", "markRoiEndButton", "clearRoiButton",
  "goToRoiStartButton", "goToRoiEndButton", "roiReadout",
  "roiStartTick", "roiEndTick", "roiSpan",
  "previousTrialButton", "nextTrialButton", "trialProgress",
].map((id) => [id, document.getElementById(id)]));

const TILE_SIZE_STORAGE_KEY = "pose-video-review:tile-size";
const { frameForMediaTime: indexedFrameForMediaTime, frameSeekTime: indexedFrameSeekTime } = FrameTiming;
const { filterTrials } = TrialFilters;

async function getJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function setStatus(message, error = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", error);
}

function trialStatusLabel(trial) {
  if (trial.saveStatus === "saved") return "Saved";
  return "Unsaved";
}

function roiStatusFromPayload(payload) {
  if (!payload || payload.enabled === false) return null;
  if (payload.start && payload.end) return "complete";
  if (payload.start) return "start";
  if (payload.end) return "end";
  return "none";
}

function trialOptionLabel(trial) {
  if (trial.roiStatus === "complete") return `✓ ${trial.id}`;
  if (trial.roiStatus === "start") return `start only · ${trial.id}`;
  if (trial.roiStatus === "end") return `end only · ${trial.id}`;
  if (trial.roiStatus === "none") return trial.id;
  return `${trial.id} · ${trial.cameraCount} cameras · ${trialStatusLabel(trial)}`;
}

function filteredTrials() {
  return filterTrials(state.trials, els.trialTypeFilter.value, els.saveFilter.value);
}

function updateTrialProgress() {
  const trials = filteredTrials();
  const index = trials.findIndex((trial) => trial.id === els.trialSelect.value);
  const position = trials.length ? `${Math.max(index, 0) + 1} / ${trials.length}` : "0 / 0";
  const marked = trials.filter((trial) => trial.roiStatus === "complete").length;
  const partial = trials.filter((trial) => trial.roiStatus === "start" || trial.roiStatus === "end").length;
  let text = position;
  if (trials.some((trial) => trial.roiStatus)) {
    text += ` · ${marked} ROI marked`;
    if (partial) text += `, ${partial} incomplete`;
  }
  els.trialProgress.textContent = text;
  const single = trials.length < 2;
  els.previousTrialButton.disabled = single;
  els.nextTrialButton.disabled = single;
}

function rememberRoiStatus(trialId, payload) {
  const status = roiStatusFromPayload(payload);
  const trial = state.trials.find((item) => item.id === trialId);
  if (trial && status) trial.roiStatus = status;
  renderTrialOptions(trialId);
}

function shiftTrial(direction) {
  const trials = filteredTrials();
  if (trials.length < 2) return;
  const index = trials.findIndex((trial) => trial.id === els.trialSelect.value);
  const start = index < 0 ? 0 : index;
  const next = trials[(start + direction + trials.length) % trials.length];
  els.trialSelect.value = next.id;
  loadTrial(next.id).catch((error) => setStatus(error.message, true));
}

function renderTrialOptions(preferredId = els.trialSelect.value) {
  const trials = filteredTrials();
  const typeCounts = { dynamic: 0, neutral: 0 };
  for (const trial of state.trials) typeCounts[trial.trialType] += 1;
  els.trialTypeFilter.options[0].textContent = `Dynamic (${typeCounts.dynamic})`;
  els.trialTypeFilter.options[1].textContent = `Neutral (${typeCounts.neutral})`;
  els.trialTypeFilter.options[2].textContent = `All trials (${state.trials.length})`;

  const typeTrials = filterTrials(state.trials, els.trialTypeFilter.value, "all");
  const counts = { unsaved: 0, saved: 0 };
  for (const trial of typeTrials) counts[trial.saveStatus] += 1;
  els.saveFilter.options[0].textContent = `All (${typeTrials.length})`;
  els.saveFilter.options[1].textContent = `Unsaved (${counts.unsaved})`;
  els.saveFilter.options[2].textContent = `Saved (${counts.saved})`;
  els.trialSelect.innerHTML = "";
  for (const trial of trials) {
    const option = document.createElement("option");
    option.value = trial.id;
    option.textContent = trialOptionLabel(trial);
    els.trialSelect.append(option);
  }
  if (trials.some((trial) => trial.id === preferredId)) els.trialSelect.value = preferredId;
  els.trialSelect.disabled = trials.length === 0;
  updateTrialProgress();
  return els.trialSelect.value || null;
}

function showEmptyFilter() {
  state.loadRevision += 1;
  pauseVideos();
  state.entries = [];
  els.cameraGrid.innerHTML = "";
  els.saveOffsetsButton.disabled = true;
  els.resetOffsetsButton.disabled = true;
  const parts = [
    els.saveFilter.value === "all" ? "" : els.saveFilter.value,
    els.trialTypeFilter.value === "all" ? "" : els.trialTypeFilter.value,
    "trials",
  ];
  setStatus(`No ${parts.filter(Boolean).join(" ")}`);
  updateTrialProgress();
}

function setTileSize(value) {
  const size = Math.max(260, Math.min(900, Math.round(Number(value) || 430)));
  document.documentElement.style.setProperty("--tile-width", `${size}px`);
  els.tileSize.value = String(size);
  els.tileSizeValue.textContent = `${size} px`;
  localStorage.setItem(TILE_SIZE_STORAGE_KEY, String(size));
  requestAnimationFrame(render);
}

function loadOffset(entry) {
  const saved = Number(entry.savedOffsetFrames);
  if (Number.isFinite(saved)) return Math.round(saved);
  return 0;
}

function offsetFrames(entry) {
  return state.offsets.get(entry.id) || 0;
}

function frameForMediaTime(entry, mediaTime) {
  return indexedFrameForMediaTime(entry.frameTimes, entry.fps, mediaTime);
}

function baseFrameAt(entry, elapsed = state.elapsed) {
  return Math.max(0, Math.min(entry.numFrames - 1, frameForMediaTime(entry, elapsed)));
}

function frameAt(entry, elapsed = state.elapsed) {
  const frame = baseFrameAt(entry, elapsed) + offsetFrames(entry);
  return Math.max(0, Math.min(entry.numFrames - 1, frame));
}

function frameSeekTime(entry, frame, attempt = 0) {
  return indexedFrameSeekTime(entry.frameTimes, entry.fps, frame, attempt);
}

function referenceStep() {
  if (!state.entries.length) return 1 / 30;
  const entry = state.entries[0];
  if (!entry.frameTimes?.length) return 1 / entry.fps;
  const frame = baseFrameAt(entry);
  if (frame + 1 < entry.frameTimes.length) return entry.frameTimes[frame + 1] - entry.frameTimes[frame];
  return frame > 0 ? entry.frameTimes[frame] - entry.frameTimes[frame - 1] : 1 / entry.fps;
}

function videoFor(entry) {
  return document.querySelector(`video[data-entry-id="${entry.id}"]`);
}

function overlayFor(entry) {
  return document.querySelector(`canvas[data-entry-id="${entry.id}"]`);
}

function setVerification(entry, status) {
  state.verification.set(entry.id, status);
  const label = document.querySelector(`[data-frame-label="${entry.id}"]`);
  if (label) label.className = `frame-label ${status}`;
}

function updateFrameLabel(entry, requestedFrame, displayedFrame) {
  const label = document.querySelector(`[data-frame-label="${entry.id}"]`);
  if (!label) return;
  const status = state.verification.get(entry.id) || "unverified";
  const timestamp = entry.frameTimes?.[displayedFrame] ?? displayedFrame / entry.fps;
  if (status === "verified") label.textContent = `frame ${displayedFrame} ✓ · ${timestamp.toFixed(3)} s`;
  else if (status === "seeking") label.textContent = `seeking frame ${requestedFrame}…`;
  else if (status === "mismatch") label.textContent = `frame ${displayedFrame} ≠ requested ${requestedFrame}`;
  else if (status === "live") label.textContent = `frame ${displayedFrame} · live`;
  else label.textContent = `frame ${displayedFrame} · unverified`;
}

function seekEntry(entry, video, force = false, verify = !state.playing) {
  if (!video || video.readyState === HTMLMediaElement.HAVE_NOTHING) return;
  const targetFrame = frameAt(entry);
  const targetTime = frameSeekTime(entry, targetFrame);
  const supportsVerification = typeof video.requestVideoFrameCallback === "function";
  if (verify && supportsVerification) {
    if (state.presentedFrames.get(entry.id) === targetFrame && !video.seeking) {
      state.pendingFrames.delete(entry.id);
      setVerification(entry, "verified");
    } else {
      state.pendingFrames.set(entry.id, { frame: targetFrame, attempt: 0 });
      setVerification(entry, "seeking");
    }
  } else if (!supportsVerification) {
    setVerification(entry, "unverified");
  }
  if (force || Math.abs(video.currentTime - targetTime) > 0.08) video.currentTime = targetTime;
}

function observeVideoFrames(entry, video) {
  if (typeof video.requestVideoFrameCallback !== "function") {
    setVerification(entry, "unverified");
    return;
  }
  const observe = (_now, metadata) => {
    if (!video.isConnected) return;
    const actualFrame = frameForMediaTime(entry, metadata.mediaTime);
    state.presentedFrames.set(entry.id, actualFrame);
    const pending = state.pendingFrames.get(entry.id);
    if (state.playing) {
      state.pendingFrames.delete(entry.id);
      setVerification(entry, "live");
    } else if (pending && !video.seeking) {
      if (actualFrame === pending.frame) {
        state.pendingFrames.delete(entry.id);
        setVerification(entry, "verified");
      } else if (pending.attempt < 2) {
        pending.attempt += 1;
        setVerification(entry, "seeking");
        video.currentTime = frameSeekTime(entry, pending.frame, pending.attempt);
      } else {
        state.pendingFrames.delete(entry.id);
        setVerification(entry, "mismatch");
      }
    }
    renderEntry(entry);
    video.requestVideoFrameCallback(observe);
  };
  video.requestVideoFrameCallback(observe);
}

function pauseVideos() {
  state.playing = false;
  els.playButton.textContent = "Play";
  if (state.animationFrame !== null) cancelAnimationFrame(state.animationFrame);
  state.animationFrame = null;
  for (const entry of state.entries) videoFor(entry)?.pause();
}

function seekVideos(force = false, verify = !state.playing) {
  for (const entry of state.entries) {
    const video = videoFor(entry);
    seekEntry(entry, video, force, verify);
  }
}

function resizeOverlay(entry) {
  const canvas = overlayFor(entry);
  const video = videoFor(entry);
  if (!canvas || !video) return null;
  const bounds = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(bounds.width * dpr));
  const height = Math.max(1, Math.round(bounds.height * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const containScale = Math.min(width / entry.width, height / entry.height);
  const contentWidth = entry.width * containScale;
  const contentHeight = entry.height * containScale;
  return {
    canvas,
    contentWidth,
    contentHeight,
    offsetX: (width - contentWidth) / 2,
    offsetY: (height - contentHeight) / 2,
  };
}

function drawPose(ctx, payload, frame, scaleX, scaleY) {
  if (!payload?.available || !payload.frames?.[frame]) return;
  const points = payload.frames[frame];
  ctx.lineWidth = Math.max(2, 2.5 / Math.min(scaleX, scaleY));
  ctx.strokeStyle = "#70f0b1";
  ctx.fillStyle = "#b5ffd8";
  for (const [a, b] of payload.edges || []) {
    if (!points[a] || !points[b]) continue;
    ctx.beginPath();
    ctx.moveTo(points[a][0], points[a][1]);
    ctx.lineTo(points[b][0], points[b][1]);
    ctx.stroke();
  }
  const radius = Math.max(2.5, 4 / Math.min(scaleX, scaleY));
  for (const point of Object.values(points)) {
    ctx.beginPath();
    ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderEntry(entry) {
  const requestedFrame = frameAt(entry);
  const displayedFrame = state.presentedFrames.get(entry.id) ?? requestedFrame;
  const poseFrame = displayedFrame + (entry.poseFrameOffset || 0);
  updateFrameLabel(entry, requestedFrame, displayedFrame);
  const cameraSlider = document.querySelector(`input[data-offset-input="${entry.id}"]`);
  if (cameraSlider) cameraSlider.value = String(requestedFrame);
  updateOffsetLabel(entry);
  const overlay = resizeOverlay(entry);
  if (!overlay) return;
  const { canvas, contentWidth, contentHeight, offsetX, offsetY } = overlay;
  const poseScaleX = contentWidth / (entry.poseWidth || entry.width);
  const poseScaleY = contentHeight / (entry.poseHeight || entry.height);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!els.showPose.checked) return;
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(poseScaleX, poseScaleY);
  drawPose(ctx, state.poses.get(entry.id), poseFrame, poseScaleX, poseScaleY);
  ctx.restore();
}

function sharedFrame() {
  if (!state.entries.length) return 0;
  return baseFrameAt(state.entries[0]);
}

function formatRoiBound(mark) {
  if (!mark) return "—";
  return `${Number(mark.time_s).toFixed(3)} s · frame ${mark.frame}`;
}

function paintRoiBound(button, mark, emptyLabel, filledPrefix, setClass) {
  const marked = Boolean(mark);
  button.classList.toggle(setClass, marked);
  button.textContent = marked ? `${filledPrefix} ${Number(mark.time_s).toFixed(3)} s` : emptyLabel;
  button.title = marked ? `Marked at ${formatRoiBound(mark)}. Click to replace it.` : emptyLabel;
}

function placeRoiTick(tick, mark) {
  if (!mark || !(state.duration > 0)) {
    tick.hidden = true;
    return;
  }
  const fraction = Math.max(0, Math.min(1, Number(mark.time_s) / state.duration));
  tick.hidden = false;
  tick.style.left = `${fraction * 100}%`;
  tick.title = `Go to ${formatRoiBound(mark)}`;
}

function placeRoiSpan(start, end) {
  if (!start || !end || !(state.duration > 0)) {
    els.roiSpan.hidden = true;
    return;
  }
  const left = Math.max(0, Math.min(Number(start.time_s), Number(end.time_s)));
  const right = Math.min(state.duration, Math.max(Number(start.time_s), Number(end.time_s)));
  els.roiSpan.hidden = false;
  els.roiSpan.style.left = `${(left / state.duration) * 100}%`;
  els.roiSpan.style.width = `${((right - left) / state.duration) * 100}%`;
}

function renderRoiReadout() {
  const mark = state.roiMarks || {};
  els.roiReadout.textContent = `ROI start ${formatRoiBound(mark.start)} · end ${formatRoiBound(mark.end)}`;
  paintRoiBound(els.markRoiStartButton, mark.start, "Mark ROI start", "Start", "roi-set-start");
  paintRoiBound(els.markRoiEndButton, mark.end, "Mark ROI end", "End", "roi-set-end");
  placeRoiTick(els.roiStartTick, mark.start);
  placeRoiTick(els.roiEndTick, mark.end);
  placeRoiSpan(mark.start, mark.end);
  els.goToRoiStartButton.disabled = !mark.start;
  els.goToRoiEndButton.disabled = !mark.end;
  paintRoiFrameCue();
}

function seekToRoiMark(mark) {
  if (!mark || !state.entries.length) return;
  pauseVideos();
  const reference = state.entries[0];
  const frame = Number(mark.frame);
  const elapsed = reference.frameTimes?.[frame] ?? mark.time_s;
  setElapsed(elapsed, true);
}

function paintRoiFrameCue() {
  const mark = state.roiMarks || {};
  const frame = state.entries.length ? sharedFrame() : null;
  const onStart = frame !== null && mark.start && Number(mark.start.frame) === frame;
  const onEnd = frame !== null && mark.end && Number(mark.end.frame) === frame;
  const label = onStart && onEnd
    ? "ROI Start (IC) · ROI End (TO)"
    : onStart
      ? "ROI Start (IC)"
      : onEnd
        ? "ROI End (TO)"
        : "";
  for (const tile of els.cameraGrid.querySelectorAll(".camera-tile")) {
    tile.classList.toggle("roi-frame-start", Boolean(label) && onStart && !onEnd);
    tile.classList.toggle("roi-frame-end", Boolean(label) && onEnd && !onStart);
    tile.classList.toggle("roi-frame-both", Boolean(onStart && onEnd));
    const badge = tile.querySelector(".roi-frame-badge");
    if (!badge) continue;
    badge.hidden = !label;
    badge.textContent = label;
  }
}

function setRoiControlsVisible(enabled) {
  for (const element of document.querySelectorAll("[data-roi-control]")) {
    element.hidden = !enabled;
  }
}

async function refreshRoiMarks(trialId = els.trialSelect.value) {
  if (!trialId) return;
  const payload = await getJson(`/api/roi-marks?id=${encodeURIComponent(trialId)}`);
  state.roiMarks = payload;
  setRoiControlsVisible(payload.enabled);
  const canMark = payload.enabled && state.entries.length > 0;
  els.markRoiStartButton.disabled = !canMark;
  els.markRoiEndButton.disabled = !canMark;
  els.clearRoiButton.disabled = !canMark || (!payload.start && !payload.end);
  renderRoiReadout();
}

function render() {
  els.timeSlider.value = String(state.elapsed);
  els.currentTime.textContent = `${state.elapsed.toFixed(3)} s`;
  renderRoiReadout();
  for (const entry of state.entries) renderEntry(entry);
}

function setElapsed(value, seek = true) {
  state.elapsed = Math.max(0, Math.min(state.duration, Number(value) || 0));
  if (seek) seekVideos(true);
  render();
}

function stepSharedFrame(direction) {
  if (!state.entries.length) return;
  const reference = state.entries[0];
  const current = baseFrameAt(reference);
  const target = Math.max(0, Math.min(reference.numFrames - 1, current + direction));
  const elapsed = reference.frameTimes?.[target] ?? target / reference.fps;
  setElapsed(elapsed, true);
}

function updateOffsetLabel(entry) {
  const frame = frameAt(entry);
  const baseFrame = baseFrameAt(entry);
  const frames = offsetFrames(entry);
  const output = document.querySelector(`[data-offset-label="${entry.id}"]`);
  if (output) {
    const sign = frames > 0 ? "+" : "";
    const baseTime = entry.frameTimes?.[baseFrame] ?? baseFrame / entry.fps;
    const frameTime = entry.frameTimes?.[frame] ?? frame / entry.fps;
    const seconds = frameTime - baseTime;
    output.textContent = `frame ${frame} · offset ${sign}${frames} f (${seconds >= 0 ? "+" : ""}${seconds.toFixed(3)} s)`;
  }
}

function setCameraFrame(entry, value) {
  const selectedFrame = Math.round(Number(value) || 0);
  const sharedFrame = baseFrameAt(entry);
  const frames = selectedFrame - sharedFrame;
  state.offsets.set(entry.id, frames);
  els.saveOffsetsButton.disabled = false;
  els.saveOffsetsButton.classList.add("unsaved");
  updateOffsetLabel(entry);
  seekVideos(true);
  renderEntry(entry);
}

function setFocusedTile(tile, focused) {
  document.querySelector(".camera-tile.focused")?.classList.remove("focused");
  if (focused) tile.classList.add("focused");
  document.body.classList.toggle("tile-focused", focused);
  requestAnimationFrame(render);
}

function buildTiles() {
  document.body.classList.remove("tile-focused");
  els.cameraGrid.innerHTML = "";
  for (const entry of state.entries) {
    state.offsets.set(entry.id, loadOffset(entry));
    state.verification.set(
      entry.id,
      typeof HTMLVideoElement.prototype.requestVideoFrameCallback === "function" ? "seeking" : "unverified",
    );
    const tile = document.createElement("article");
    tile.className = "camera-tile";

    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent = entry.camera;
    const frameLabel = document.createElement("span");
    frameLabel.dataset.frameLabel = entry.id;
    frameLabel.className = "frame-label seeking";
    header.append(title, frameLabel);

    const stage = document.createElement("div");
    stage.className = "video-stage";
    stage.style.aspectRatio = `${entry.width} / ${entry.height}`;
    stage.tabIndex = 0;
    stage.setAttribute("role", "button");
    stage.setAttribute("aria-label", `Enlarge ${entry.camera}`);
    stage.title = "Click to enlarge this camera; click again or press Escape to close";
    const toggleFocus = () => setFocusedTile(tile, !tile.classList.contains("focused"));
    stage.addEventListener("click", toggleFocus);
    stage.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        toggleFocus();
      }
    });
    const video = document.createElement("video");
    video.dataset.entryId = entry.id;
    video.src = `/media?id=${encodeURIComponent(entry.id)}`;
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;
    video.disablePictureInPicture = true;
    video.setAttribute("aria-label", `${entry.camera} video`);
    video.addEventListener("loadedmetadata", () => {
      observeVideoFrames(entry, video);
      seekEntry(entry, video, true, true);
    });
    video.addEventListener("seeked", () => {
      if (typeof video.requestVideoFrameCallback !== "function") renderEntry(entry);
    });
    const overlay = document.createElement("canvas");
    overlay.dataset.entryId = entry.id;
    overlay.setAttribute("aria-hidden", "true");
    const roiBadge = document.createElement("div");
    roiBadge.className = "roi-frame-badge";
    roiBadge.hidden = true;
    roiBadge.setAttribute("aria-hidden", "true");
    stage.append(video, overlay, roiBadge);

    const alignment = document.createElement("div");
    alignment.className = "alignment-control";
    const alignmentHeader = document.createElement("div");
    const alignmentTitle = document.createElement("span");
    alignmentTitle.textContent = "Camera frame";
    const offsetLabel = document.createElement("output");
    offsetLabel.dataset.offsetLabel = entry.id;
    alignmentHeader.append(alignmentTitle, offsetLabel);
    const sliderRow = document.createElement("div");
    sliderRow.className = "offset-slider-row";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.dataset.offsetInput = entry.id;
    slider.min = "0";
    slider.max = String(entry.numFrames - 1);
    slider.step = "1";
    slider.value = String(frameAt(entry));
    slider.addEventListener("input", () => {
      pauseVideos();
      setCameraFrame(entry, slider.value);
    });
    sliderRow.append(slider);
    alignment.append(alignmentHeader, sliderRow);

    const footer = document.createElement("footer");
    const metadata = document.createElement("span");
    metadata.textContent = `${entry.width}×${entry.height} · ${entry.numFrames} indexed frames`;
    const poseState = document.createElement("span");
    poseState.className = "pose-state loading";
    poseState.dataset.poseState = entry.id;
    poseState.textContent = "Checking pose…";
    footer.append(metadata, poseState);

    tile.append(header, stage, alignment, footer);
    els.cameraGrid.append(tile);
    updateOffsetLabel(entry);
  }
}

function updatePoseState(entry, payload) {
  const badge = document.querySelector(`[data-pose-state="${entry.id}"]`);
  if (!badge) return;
  badge.classList.remove("loading", "available", "missing");
  if (payload.available) {
    badge.classList.add("available");
    badge.textContent = `${entry.poseType} · ${payload.frames.length} frames`;
  } else {
    badge.classList.add("missing");
    badge.textContent = "No pose file";
    badge.title = payload.message || "No pose data is available for this camera.";
  }
}

async function loadTrial(id) {
  const revision = ++state.loadRevision;
  pauseVideos();
  setStatus("Loading trial…");
  const payload = await getJson(`/api/trial?id=${encodeURIComponent(id)}`);
  if (revision !== state.loadRevision) return;
  state.entries = payload.entries;
  state.elapsed = 0;
  state.duration = Math.min(...state.entries.map((entry) => entry.durationSeconds));
  state.poses.clear();
  state.offsets.clear();
  state.presentedFrames.clear();
  state.pendingFrames.clear();
  state.verification.clear();
  buildTiles();
  els.saveOffsetsButton.disabled = false;
  els.saveOffsetsButton.classList.remove("unsaved");
  els.resetOffsetsButton.disabled = false;
  els.timeSlider.max = String(state.duration);
  els.timeSlider.step = "any";
  els.duration.textContent = `${state.duration.toFixed(3)} s`;
  seekVideos(true);
  try { await refreshRoiMarks(id); }
  catch (error) { setStatus(error.message, true); }
  seekVideos(true);
  render();

  const poseResults = await Promise.all(state.entries.map(async (entry) => {
    try { return [entry, await getJson(`/api/poses?id=${encodeURIComponent(entry.id)}`)]; }
    catch (error) { return [entry, { available: false, message: error.message, frames: [] }]; }
  }));
  if (revision !== state.loadRevision) return;
  for (const [entry, poses] of poseResults) {
    state.poses.set(entry.id, poses);
    updatePoseState(entry, poses);
  }
  const available = poseResults.filter(([, poses]) => poses.available).length;
  setStatus(`${state.entries.length} cameras · pose data for ${available}`);
}

function stopAtEnd() {
  pauseVideos();
  state.elapsed = state.duration;
  seekVideos(true);
  render();
}

function tick(timestamp) {
  if (!state.playing) return;
  const rate = Number(els.speedSelect.value);
  state.elapsed = state.playbackStartElapsed + (timestamp - state.playbackStartedAt) / 1000 * rate;
  if (state.elapsed >= state.duration - referenceStep() / 2) {
    if (els.loopPlayback.checked) {
      setElapsed(0, true);
      state.playbackStartElapsed = 0;
      state.playbackStartedAt = timestamp;
    } else {
      stopAtEnd();
      return;
    }
  }
  if (timestamp - state.lastSyncAt > 500) {
    seekVideos(false);
    state.lastSyncAt = timestamp;
  }
  render();
  state.animationFrame = requestAnimationFrame(tick);
}

async function startPlayback() {
  if (state.elapsed >= state.duration - referenceStep() / 2) setElapsed(0, true);
  const rate = Number(els.speedSelect.value);
  state.pendingFrames.clear();
  for (const entry of state.entries) setVerification(entry, "live");
  seekVideos(true, false);
  for (const entry of state.entries) videoFor(entry).playbackRate = rate;
  try {
    await Promise.all(state.entries.map((entry) => videoFor(entry).play()));
    state.playing = true;
    els.playButton.textContent = "Pause";
    state.playbackStartElapsed = state.elapsed;
    state.playbackStartedAt = performance.now();
    state.lastSyncAt = performance.now();
    state.animationFrame = requestAnimationFrame(tick);
  } catch (error) {
    pauseVideos();
    setStatus(`Playback failed: ${error.message}`, true);
  }
}

function resetOffsets() {
  if (!state.entries.length) return;
  for (const entry of state.entries) state.offsets.set(entry.id, loadOffset(entry));
  els.saveOffsetsButton.classList.remove("unsaved");
  seekVideos(true);
  render();
  setStatus("Restored camera offsets");
}

async function saveOffsets() {
  els.saveOffsetsButton.disabled = true;
  const offsets = state.entries.map((entry) => ({
    id: entry.id,
    offset_frames: offsetFrames(entry),
  }));
  try {
    const result = await postJson("/api/offsets", { offsets });
    for (const entry of state.entries) {
      entry.savedOffsetFrames = offsetFrames(entry);
      entry.hasSavedOffset = true;
    }
    const currentTrial = state.trials.find((trial) => trial.id === els.trialSelect.value);
    if (currentTrial) {
      currentTrial.savedCameraCount = currentTrial.cameraCount;
      currentTrial.saveStatus = "saved";
    }
    els.saveOffsetsButton.classList.remove("unsaved");
    setStatus(`Saved ${result.saved} camera offsets`);
    const nextId = renderTrialOptions(currentTrial?.id);
    if (nextId && nextId !== currentTrial?.id) await loadTrial(nextId);
    else if (!nextId) showEmptyFilter();
  } catch (error) {
    els.saveOffsetsButton.disabled = false;
    setStatus(`Could not save offsets: ${error.message}`, true);
  }
}

els.trialSelect.addEventListener("change", () => loadTrial(els.trialSelect.value).catch((error) => setStatus(error.message, true)));
els.previousTrialButton.addEventListener("click", () => shiftTrial(-1));
els.nextTrialButton.addEventListener("click", () => shiftTrial(1));
function applyTrialFilters() {
  const nextId = renderTrialOptions();
  if (nextId) loadTrial(nextId).catch((error) => setStatus(error.message, true));
  else showEmptyFilter();
}

els.trialTypeFilter.addEventListener("change", applyTrialFilters);
els.saveFilter.addEventListener("change", applyTrialFilters);
els.saveOffsetsButton.addEventListener("click", saveOffsets);
els.resetOffsetsButton.addEventListener("click", resetOffsets);
function goToMarkedRoi(bound) {
  seekToRoiMark(state.roiMarks?.[bound]);
}
els.roiStartTick.addEventListener("click", () => goToMarkedRoi("start"));
els.roiEndTick.addEventListener("click", () => goToMarkedRoi("end"));
els.goToRoiStartButton.addEventListener("click", () => goToMarkedRoi("start"));
els.goToRoiEndButton.addEventListener("click", () => goToMarkedRoi("end"));
els.tileSize.addEventListener("input", () => setTileSize(els.tileSize.value));
async function markRoi(bound) {
  pauseVideos();
  try {
    state.roiMarks = await postJson("/api/roi-marks", {
      id: els.trialSelect.value,
      bound,
      time_s: state.elapsed,
      frame: sharedFrame(),
    });
    els.clearRoiButton.disabled = false;
    renderRoiReadout();
    rememberRoiStatus(els.trialSelect.value, state.roiMarks);
    setStatus(`Marked ROI ${bound}`);
  } catch (error) {
    setStatus(error.message, true);
  }
}
els.markRoiStartButton.addEventListener("click", () => markRoi("start"));
els.markRoiEndButton.addEventListener("click", () => markRoi("end"));
els.clearRoiButton.addEventListener("click", async () => {
  try {
    state.roiMarks = await postJson("/api/roi-marks", { id: els.trialSelect.value, clear: true });
    els.clearRoiButton.disabled = true;
    renderRoiReadout();
    rememberRoiStatus(els.trialSelect.value, state.roiMarks);
    setStatus("Cleared ROI");
  } catch (error) {
    setStatus(error.message, true);
  }
});
els.showPose.addEventListener("change", render);
els.timeSlider.addEventListener("input", () => {
  pauseVideos();
  setElapsed(els.timeSlider.value, true);
});
els.stepBackButton.addEventListener("click", () => {
  pauseVideos();
  stepSharedFrame(-1);
});
els.stepForwardButton.addEventListener("click", () => {
  pauseVideos();
  stepSharedFrame(1);
});
els.playButton.addEventListener("click", () => {
  if (state.playing) {
    pauseVideos();
    seekVideos(true, true);
  }
  else startPlayback();
});
els.speedSelect.addEventListener("change", () => {
  const rate = Number(els.speedSelect.value);
  for (const entry of state.entries) {
    const video = videoFor(entry);
    if (video) video.playbackRate = rate;
  }
});
window.addEventListener("resize", render);
function hotkeysBlocked(target) {
  if (target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) return true;
  if (target instanceof HTMLInputElement) {
    const type = String(target.type || "text").toLowerCase();
    return type !== "range" && type !== "checkbox";
  }
  return false;
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.querySelector(".camera-tile.focused")) {
    setFocusedTile(document.querySelector(".camera-tile.focused"), false);
    return;
  }
  if (hotkeysBlocked(event.target)) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    if (event.key === "ArrowLeft") els.stepBackButton.click();
    else els.stepForwardButton.click();
    return;
  }
  if (event.key === " " && !(event.target instanceof HTMLButtonElement)) {
    event.preventDefault();
    els.playButton.click();
  }
});

setTileSize(localStorage.getItem(TILE_SIZE_STORAGE_KEY) || els.tileSize.value);

async function hydrateRoiStatus(trials) {
  if (!state.roiMarks.enabled || trials.every((trial) => trial.roiStatus)) return;
  await Promise.all(trials.map(async (trial) => {
    if (trial.roiStatus) return;
    const marks = await getJson(`/api/roi-marks?id=${encodeURIComponent(trial.id)}`);
    trial.roiStatus = roiStatusFromPayload(marks);
  }));
}

getJson("/api/roi-marks").then((payload) => {
  state.roiMarks = payload;
  setRoiControlsVisible(payload.enabled);
  renderRoiReadout();
  return getJson("/api/trials");
}).catch(() => {
  setRoiControlsVisible(false);
  return getJson("/api/trials");
}).then(async ({ trials }) => {
  state.trials = trials;
  if (!trials.length) throw new Error("No trials were found.");
  await hydrateRoiStatus(trials);
  const firstId = renderTrialOptions();
  if (firstId) return loadTrial(firstId);
  showEmptyFilter();
  return undefined;
}).catch((error) => setStatus(error.message, true));
