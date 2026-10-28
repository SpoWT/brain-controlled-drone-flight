// Neural Ride page controller: navigation, headset, rides, and the drive mode.
import {PitWall, SignalCheck, api} from "./panels.js";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const STEPS = ["headset", "signal", "ride", "train", "drive"];
const FLUSH_MS = 400;

const state = {
  view: "headset",
  status: null,
  minutes: 8,
  ride: null,
  queue: [],
  flushing: null,
  mind: {state: "STOP", p_go: 0},
  mindTimer: null,
  connectRequest: 0,
  mindPolling: 0,
};

function toast(message, isError = false) {
  const element = document.createElement("div");
  element.className = `toast${isError ? " is-error" : ""}`;
  element.textContent = message;
  $("#toasts").append(element);
  window.setTimeout(() => element.remove(), isError ? 9000 : 5000);
}

const signal = new SignalCheck($("#channels"), {headline: $("#signal-headline"), lede: $("#signal-lede")});
const pitWall = new PitWall({toast});

// ---------------------------------------------------------------- navigation
function connected() {
  return state.status?.stream?.state === "streaming";
}

function show(view) {
  const target = view === "ride" ? "signal" : view;
  state.view = target;
  for (const section of $$("[data-view]")) section.hidden = section.dataset.view !== target;
  const index = STEPS.indexOf(view === "ride" ? "signal" : target);
  $$(".step").forEach((step, i) => {
    step.classList.toggle("now", i === index);
    step.classList.toggle("done", i < index);
  });
  $$(".seg [data-mode]").forEach((tab) => tab.setAttribute("aria-selected", String((tab.dataset.mode === "drive") === (target === "drive"))));
  if (target === "signal" && connected()) signal.start(); else signal.stop();
  if (target === "train") pitWall.refresh();
  if (target === "drive") renderDrive();
}

function go(view) {
  if ((view === "signal" || view === "ride") && !connected()) {
    toast("Connect the headset or the simulator first.");
    view = "headset";
  }
  show(view);
}

// ---------------------------------------------------------------- headset
function renderStatus(status) {
  state.status = status;
  const stream = status.stream;
  const dot = $("#headset-chip .dot");
  const label = $("#headset-label");
  dot.className = "dot off";
  if (stream.state === "streaming") {
    dot.className = "dot";
    label.textContent = stream.source.kind === "simulated" ? "SIMULATOR · 200 Hz" : `GANGLION · ${stream.source.transport}`;
  } else if (stream.state === "stalled") {
    dot.className = "dot warn";
    label.textContent = "SIGNAL STALLED";
  } else if (stream.state === "error") {
    dot.className = "dot bad";
    label.textContent = "HEADSET LOST";
  } else {
    label.textContent = "NO HEADSET";
  }
}

async function refreshStatus() {
  try {
    renderStatus(await api("/api/status"));
  } catch (error) {
    $("#headset-label").textContent = "SERVER OFFLINE";
    $("#headset-chip .dot").className = "dot bad";
  }
}

async function connect(mode) {
  const button = mode === "simulated" ? $("#connect-sim") : $("#connect-ganglion");
  const request = ++state.connectRequest;  // a newer connect or STOP makes this answer stale
  button.classList.add("is-busy");
  $("#connect-note").className = "note ok";
  $("#connect-note").textContent = mode === "simulated" ? "Starting the simulator…" : "Searching for the Ganglion (up to 20 s)…";
  try {
    const stream = await api("/api/connect", {method: "POST", body: {mode}});
    if (request !== state.connectRequest) return;
    renderStatus({stream});
    $("#connect-note").textContent = "";
    show("signal");
  } catch (error) {
    if (request !== state.connectRequest) return;
    $("#connect-note").className = "note";
    $("#connect-note").textContent = error.message;
  } finally {
    button.classList.remove("is-busy");
  }
}

// ---------------------------------------------------------------- rides
function queueEvent(event) {
  state.queue.push(event);
}

async function flush() {
  if (!state.queue.length) return;
  const batch = state.queue.splice(0, 400);
  try {
    await api("/api/ride/events", {method: "POST", body: {events: batch}});
  } catch (error) {
    toast(`Events not saved: ${error.message}`, true);
  }
}

function startFlushing() {
  stopFlushing();
  state.flushing = window.setInterval(flush, FLUSH_MS);
}

function stopFlushing() {
  if (state.flushing) window.clearInterval(state.flushing);
  state.flushing = null;
}

function startMindPolling() {
  stopMindPolling();
  const polling = ++state.mindPolling;  // stopMindPolling bumps it; an answer arriving after that is dropped
  const tick = async () => {
    let mind;
    try {
      mind = await api("/api/live");
    } catch (error) {
      mind = {state: "STOP", p_go: 0, reason: "game server not answering"};
    }
    if (polling !== state.mindPolling) return;
    state.mind = mind;
    drawMind();
    state.mindTimer = window.setTimeout(tick, 100);
  };
  tick();
}

function stopMindPolling() {
  state.mindPolling += 1;
  if (state.mindTimer) window.clearTimeout(state.mindTimer);
  state.mindTimer = null;
  state.mind = {state: "STOP", p_go: 0};
}

function drawMind() {
  // Why the model is holding STOP: collecting history, a signal gap, an artifact...
  const reason = state.mind.reason || "";
  $("#mind-note").textContent = reason && reason !== "decoding" && reason !== "not driving" ? reason.toUpperCase() : "";
  const canvas = $("#mind-gauge");
  const ratio = window.devicePixelRatio || 1;
  const size = canvas.clientWidth * ratio;
  if (!size) return;
  if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, size, size);
  const p = Math.max(0, Math.min(1, state.mind.p_go || 0));
  g.lineCap = "round";
  g.lineWidth = size * 0.02;
  g.strokeStyle = state.mind.state === "GO" ? "rgba(127,214,154,0.95)" : "rgba(240,138,114,0.85)";
  g.beginPath();
  g.arc(size / 2, size / 2, size * 0.46, Math.PI * 0.8, Math.PI * 0.8 + Math.PI * 1.4 * p + 0.001);
  g.stroke();
}

async function startRide(kind) {
  if (state.ride) return;
  const ride = await loadRide();
  if (!ride) return;
  const button = kind === "drive" ? $("#start-drive") : $("#start-ride");
  button.classList.add("is-busy");
  try {
    await api("/api/ride/start", {method: "POST", body: {kind, minutes: state.minutes}});
  } catch (error) {
    toast(error.message, true);
    button.classList.remove("is-busy");
    return;
  }
  button.classList.remove("is-busy");
  signal.stop();
  document.activeElement?.blur?.();
  document.body.classList.add("is-riding");
  $("#ride").classList.toggle("is-mind", kind === "drive");
  state.ride = kind;
  startFlushing();
  if (kind === "drive") startMindPolling();
  await ride.run({
    minutes: state.minutes,
    control: kind === "drive" ? "mind" : "keys",
    mindState: () => state.mind.state,
  });
  await finishRide();
}

async function finishRide() {
  stopFlushing();
  stopMindPolling();
  await flush();
  document.body.classList.remove("is-riding");
  $("#ride").classList.remove("is-mind");
  state.ride = null;
  try {
    const manifest = await api("/api/ride/finish", {method: "POST"});
    pitWall.showRide(manifest);
  } catch (error) {
    toast(error.message, true);
  }
  show("train");
}

async function stopAll() {
  state.connectRequest += 1;
  rideInstance?.abort();
  stopFlushing();
  stopMindPolling();
  try {
    renderStatus(await api("/api/stop", {method: "POST"}));
  } catch (error) {
    toast(error.message, true);
  }
  document.body.classList.remove("is-riding");
  show("headset");
}

async function renderDrive() {
  const model = state.status?.model;
  const start = $("#start-drive");
  if (!model) {
    $("#drive-headline").textContent = "No model yet.";
    $("#drive-lede").textContent = "Record a few rides, then train. The drive unlocks once a model exists.";
    start.disabled = true;
    return;
  }
  const accuracy = Math.round(model.balanced_accuracy * 100);
  $("#drive-headline").textContent = "The model holds the pedal.";
  const origin = model.simulated ? " Trained without headset rides: drive it with the simulator." : "";
  $("#drive-lede").textContent = `Hands off the keys. Model ${model.bundle}: held-out balanced accuracy ${accuracy}% (chance 50%).${origin} Every drive is recorded too.`;
  start.disabled = !connected();
  $("#drive-hint").innerHTML = connected() ? "<kbd>esc</kbd> ends the drive" : "connect the headset first";
}

// ---------------------------------------------------------------- 3D ride
let rideInstance = null;
let rideLoading = null;

function loadRide() {
  if (rideInstance) return Promise.resolve(rideInstance);
  if (!rideLoading) {
    rideLoading = import("./ride/ride.js")
      .then(({Ride}) => {
        rideInstance = new Ride($("#ride"), {onEvent: queueEvent});
        if (new URLSearchParams(window.location.search).has("debug")) window.neuralRide = rideInstance;
        return rideInstance;
      })
      .catch((error) => {
        $("#ride-fallback").hidden = false;
        toast(`3D ride failed to start: ${error.message}`, true);
        rideLoading = null;
        return null;
      });
  }
  return rideLoading;
}

// ---------------------------------------------------------------- wiring
function bind() {
  $$(".step").forEach((step) => step.addEventListener("click", () => go(step.dataset.step)));
  $$(".seg [data-mode]").forEach((tab) => tab.addEventListener("click", () => go(tab.dataset.mode === "drive" ? "drive" : (connected() ? "signal" : "headset"))));
  $$(".length [data-minutes]").forEach((button) => button.addEventListener("click", () => {
    state.minutes = Number(button.dataset.minutes);
    $$(".length [data-minutes]").forEach((b) => b.setAttribute("aria-checked", String(b === button)));
  }));
  $("#connect-ganglion").addEventListener("click", () => connect("ganglion"));
  $("#connect-sim").addEventListener("click", () => connect("simulated"));
  $("#start-ride").addEventListener("click", () => startRide("ride"));
  $("#start-drive").addEventListener("click", () => startRide("drive"));
  $("#ride-again").addEventListener("click", () => go("signal"));
  $("#train-model").addEventListener("click", async () => {
    await pitWall.train();
    await refreshStatus();
  });
  $("#go-drive").addEventListener("click", () => go("drive"));
  $("#stop-all").addEventListener("click", stopAll);
}

async function boot() {
  bind();
  await refreshStatus();
  show(connected() ? "signal" : "headset");
  pitWall.showResult(state.status?.model || null);
  window.setInterval(refreshStatus, 1500);
  loadRide();
}

boot();
