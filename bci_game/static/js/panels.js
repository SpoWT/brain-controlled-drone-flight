// Page panels: the JSON client, the signal check and the pit wall.

// Thin JSON client. Every state change carries the page token.
const token = document.querySelector('meta[name="neural-ride-token"]').content;

export async function api(path, {method = "GET", body} = {}) {
  const options = {method, headers: {Accept: "application/json"}};
  if (method !== "GET") {
    options.headers["Content-Type"] = "application/json";
    options.headers["X-Token"] = token;
    options.body = JSON.stringify(body ?? {});
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new Error("The Neural Ride server is not answering. Is it still running?");
  }
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`Server returned HTTP ${response.status}.`);
  }
  if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload.data;
}

// Live per-channel traces and quality pills for the signal check.

const COLORS = {CLEAN: "#7fd69a", NOISY: "#f08a72", FLAT: "#f08a72", WAIT: "#8b847a"};
const WORDS = ["None", "One", "Two", "Three", "All four"];

export class SignalCheck {
  constructor(root, {headline, lede}) {
    this.root = root;
    this.headline = headline;
    this.lede = lede;
    this.rows = new Map();
    this.timer = null;
  }

  _row(name) {
    if (this.rows.has(name)) return this.rows.get(name);
    const row = document.createElement("div");
    row.className = "chan";
    const label = document.createElement("b");
    label.textContent = name;
    const canvas = document.createElement("canvas");
    const num = document.createElement("div");
    num.className = "num";
    const pill = document.createElement("span");
    pill.className = "pill";
    row.append(label, canvas, num, pill);
    this.root.append(row);
    const entry = {canvas, num, pill};
    this.rows.set(name, entry);
    return entry;
  }

  start() {
    this.stop();
    const tick = async () => {
      try {
        this.render(await api("/api/signal"));
      } catch (error) {
        this.headline.textContent = "Signal lost.";
        this.lede.textContent = error.message;
      }
      this.timer = window.setTimeout(tick, 250);
    };
    tick();
  }

  stop() {
    if (this.timer) window.clearTimeout(this.timer);
    this.timer = null;
  }

  render(data) {
    data.channels.forEach((channel, index) => {
      const row = this._row(channel.name);
      row.num.textContent = channel.rms_uv === null ? "—" : `${channel.rms_uv.toFixed(1)} µV`;
      row.pill.textContent = channel.state;
      row.pill.className = `pill ${channel.state}`;
      draw(row.canvas, data.trace[index] || [], COLORS[channel.state]);
    });
    const waiting = data.channels.some((c) => c.state === "WAIT");
    const clean = data.channels.filter((c) => c.state === "CLEAN").length;
    if (waiting) {
      this.headline.textContent = "Reading the channels…";
      this.lede.textContent = "Sit still for a few seconds while each channel settles.";
      return;
    }
    const bad = data.channels.filter((c) => c.state !== "CLEAN");
    this.headline.textContent = clean === 4 ? "All four channels are clean." : `${WORDS[clean]} of four channels ${clean === 1 ? "is" : "are"} clean.`;
    this.lede.textContent = clean === 4
      ? "Relax your jaw and shoulders. Ride whenever you are ready."
      : bad.map((c) => c.state === "FLAT"
          ? `${c.name} is flat — its lead may be off the pin.`
          : `${c.name} is noisy — work the comb through the hair until it calms.`).join(" ");
  }
}

function draw(canvas, values, color) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth * ratio;
  const height = canvas.clientHeight * ratio;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, width, height);
  if (values.length < 2) return;
  const span = Math.max(30, ...values.map(Math.abs));
  g.strokeStyle = color;
  g.lineWidth = 1.2 * ratio;
  g.beginPath();
  values.forEach((value, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height / 2 - (value / span) * (height / 2 - 2);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  });
  g.stroke();
}

// The pit wall: saved rides, include/exclude toggles, training, and results.

const $ = (selector) => document.querySelector(selector);
const MIN_RIDES = 2;

function clock(seconds) {
  const whole = Math.round(Number(seconds) || 0);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function when(id) {
  const match = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(id);
  return match ? `${match[2]}-${match[3]} ${match[4]}:${match[5]}` : id;
}

function cell(text, className = "") {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

export class PitWall {
  constructor({toast}) {
    this.toast = toast;
    this.pollTimer = null;
    this.training = false;
    $("#sessions").addEventListener("click", (event) => this._toggle(event));
  }

  // Headline for the most recent ride, shown right after it ends.
  showRide(manifest) {
    const s = manifest.summary || {};
    const kind = manifest.kind === "drive" ? "DRIVE SAVED" : "RIDE SAVED";
    $("#train-kicker").textContent = `${kind} · ${clock(manifest.duration_s)}`;
    $("#train-headline").textContent = `${s.stops ?? 0} stops,\n${s.late ?? 0} late.`;
    $("#train-headline").style.whiteSpace = "pre-line";
    const stop = s.stop_reaction_ms ? `Median stop reaction ${s.stop_reaction_ms} ms` : "No timed stop reactions";
    const go = s.go_reaction_ms ? `, back on the pedal in ${s.go_reaction_ms} ms` : "";
    $("#train-lede").textContent = `${stop}${go}. Every keypress and hazard is stored next to the raw EEG.`;
  }

  async refresh() {
    let sessions = [];
    try {
      sessions = await api("/api/sessions");
    } catch (error) {
      this.toast(error.message, true);
    }
    const body = $("#sessions");
    body.replaceChildren();
    const rides = sessions.filter((m) => m.kind === "ride");
    if (!sessions.length) {
      const row = document.createElement("tr");
      row.append(cell("No rides recorded yet.", "empty"));
      row.firstChild.colSpan = 6;
      body.append(row);
    }
    for (const manifest of sessions) {
      const s = manifest.summary || {};
      const row = document.createElement("tr");
      row.append(
        cell(`${when(manifest.session_id)}${manifest.kind === "drive" ? " · mind" : ""}${manifest.source === "simulated" ? " · sim" : ""}`),
        cell(clock(manifest.duration_s)),
        cell(String(s.stops ?? "—")),
        cell(String(s.late ?? "—")),
        cell(s.stop_reaction_ms ? `${s.stop_reaction_ms} ms` : "—"),
      );
      const use = document.createElement("td");
      if (manifest.kind === "ride") {
        const pill = document.createElement("button");
        pill.type = "button";
        pill.className = `pill ${manifest.excluded ? "OUT" : "IN"}`;
        pill.textContent = manifest.excluded ? "OUT" : "IN";
        pill.dataset.session = manifest.session_id;
        pill.dataset.excluded = String(Boolean(manifest.excluded));
        pill.title = "Click to include or exclude this ride from training";
        use.append(pill);
      } else {
        use.textContent = "test";
        use.className = "empty";
      }
      row.append(use);
      body.append(row);
    }
    if (!$("#train-kicker").textContent.includes("SAVED")) {
      $("#train-kicker").textContent = "RIDES";
      $("#train-headline").textContent = rides.length ? `${rides.length} ride${rides.length === 1 ? "" : "s"} on file.` : "No rides yet.";
    }
    $("#train-model").disabled = rides.filter((m) => !m.excluded).length < MIN_RIDES;
    $("#train-note").className = "note ok";
    if (!this.training) $("#train-note").textContent = rides.filter((m) => !m.excluded).length < MIN_RIDES ? `Training needs at least ${MIN_RIDES} rides marked IN.` : "";
    return sessions;
  }

  async _toggle(event) {
    const pill = event.target.closest("button[data-session]");
    if (!pill) return;
    pill.disabled = true;
    try {
      await api(`/api/sessions/${encodeURIComponent(pill.dataset.session)}/exclude`, {
        method: "POST", body: {excluded: pill.dataset.excluded !== "true"},
      });
    } catch (error) {
      this.toast(error.message, true);
    }
    await this.refresh();
  }

  async train() {
    const button = $("#train-model");
    button.classList.add("is-busy");
    this.training = true;
    $("#train-note").className = "note ok";
    $("#train-note").textContent = "Training…";
    try {
      await api("/api/train", {method: "POST"});
      await this._poll();
    } catch (error) {
      $("#train-note").className = "note";
      $("#train-note").textContent = error.message;
    } finally {
      this.training = false;
      button.classList.remove("is-busy");
    }
  }

  _poll() {
    return new Promise((resolve) => {
      const tick = async () => {
        let state;
        try {
          state = await api("/api/train");
        } catch (error) {
          $("#train-note").className = "note";
          $("#train-note").textContent = error.message;
          resolve();
          return;
        }
        if (state.state === "running") {
          $("#train-note").textContent = state.progress || "Training…";
          this.pollTimer = window.setTimeout(tick, 500);
          return;
        }
        const report = state.state === "done" ? state.report : state.latest;
        if (state.state === "failed") {
          $("#train-note").className = "note";
          $("#train-note").textContent = state.error;
        } else if (state.state === "done") {
          $("#train-note").textContent = report.simulated
            ? `Saved ${report.folder}. Trained without headset rides: for the DRIVE step only, never flown.`
            : `Saved ${report.folder}. After a restart the flying website offers it; Auto flies the best held-out bundle.`;
        }
        this.showResult(report);
        resolve();
      };
      tick();
    });
  }

  showResult(report) {
    const panel = $("#result");
    if (!report) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    const accuracy = Math.round(report.balanced_accuracy * 100);
    $("#result-method").textContent = `${report.method.toUpperCase()} · BALANCED ACCURACY`;
    $("#result-accuracy").replaceChildren(String(accuracy), Object.assign(document.createElement("small"), {textContent: "%"}));
    const meter = $("#result-meter");
    meter.style.width = `${accuracy}%`;
    meter.classList.toggle("weak", accuracy < 60);
    const windows = report.windows || {};
    $("#result-detail").textContent = `${report.model || "XGBoost"} · chance is 50% · ${report.sessions.length} rides · ${report.bundle}${report.simulated ? " · simulator" : ""}`;
    this.showBoard(report.leaderboard || []);
    $("#result-recall").textContent = `STOP recall ${Math.round(report.stop_recall * 100)}% · GO recall ${Math.round(report.go_recall * 100)}%\n${(windows.GO || 0) + (windows.STOP || 0)} windows · 1 s · 0.5 s step`;
    $("#result-recall").style.whiteSpace = "pre-line";
  }

  // Every model trained, best first; the first one is the saved bundle.
  showBoard(rows) {
    const items = rows.map((row, index) => {
      const item = document.createElement("li");
      const score = "balanced_accuracy" in row ? `${Math.round(row.balanced_accuracy * 100)}%` : "failed";
      item.className = index === 0 && "balanced_accuracy" in row ? "best" : "balanced_accuracy" in row ? "" : "failed";
      item.title = row.error || "";
      item.append(Object.assign(document.createElement("span"), {textContent: row.label}),
                  Object.assign(document.createElement("span"), {textContent: score}));
      return item;
    });
    $("#result-board").replaceChildren(...items);
  }
}
