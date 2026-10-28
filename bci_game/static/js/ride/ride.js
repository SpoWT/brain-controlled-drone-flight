// The ride: renderer, world, input, wordless HUD, and the event stream that
// becomes the EEG labels.  Hold ArrowUp / W / Space to drive; let go to stop.
import * as THREE from "../../vendor/three.module.min.js";
import {Director} from "./director.js";
import {Effects, RideAudio} from "./effects.js";
import {Road} from "./road.js";
import {Environment, Scenery} from "./world.js";
import * as tex from "./assets.js";
import {MAX_SPEED, Vehicle} from "./vehicle.js";

const DRIVE_KEYS = new Set(["ArrowUp", "KeyW", "Space"]);
const IDLE_SPEED = 11;
const SPEED_SAMPLE_S = 0.5;

const now = () => Date.now();  // same wall clock as the EEG timestamps

function materials(textures) {
  return {
    bark: new THREE.MeshStandardMaterial({map: textures.bark, roughness: 0.95}),
    conifer: new THREE.MeshStandardMaterial({color: 0xffffff, roughness: 0.9}),
    heroConifer: new THREE.MeshStandardMaterial({color: 0x3a5a2c, roughness: 0.9}),
    broadleaf: new THREE.MeshStandardMaterial({color: 0xffffff, roughness: 0.85}),
    bush: new THREE.MeshStandardMaterial({color: 0xffffff, roughness: 1}),
    rock: new THREE.MeshStandardMaterial({map: textures.rock, color: 0x9aa0a8, roughness: 0.95}),
    paint: new THREE.MeshStandardMaterial({color: 0x1b2631, metalness: 0.15, roughness: 0.62, envMapIntensity: 0.3}),
    trim: new THREE.MeshStandardMaterial({color: 0x121314, roughness: 0.82}),
    paintLine: new THREE.MeshStandardMaterial({color: 0xdedbd2, roughness: 0.7, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4}),
    crossRoad: new THREE.MeshStandardMaterial({color: 0x47484a, roughness: 0.95, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1}),
  };
}

export class Ride {
  constructor(container, {onEvent = () => {}, preset} = {}) {
    this.container = container;
    this.onEvent = onEvent;
    this.mode = "idle";
    this.paused = false; // debug freeze-frame only
    this.keys = new Set();
    this.listeners = [];

    this.renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: "high-performance"});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    container.prepend(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.05, 5000);
    const anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    this.textures = {
      asphalt: tex.asphaltTexture(anisotropy), gravel: tex.gravelTexture(anisotropy),
      grass: tex.grassTexture(anisotropy), bark: tex.barkTexture(), rock: tex.rockTexture(),
      glow: tex.glowTexture(), dust: tex.dustTexture(),
    };
    this.materials = materials(this.textures);
    const presetName = preset || (Math.random() < 0.65 ? "golden" : "overcast");
    this.environment = new Environment(this.scene, this.renderer, this.textures, presetName);
    this.road = new Road(this.textures);
    this.scene.add(this.road.group);
    this.scenery = new Scenery(this.scene, this.materials);
    this.car = new Vehicle(this.scene, this.camera, this.materials);
    this.audio = new RideAudio();
    this.flash = container.querySelector("[data-ride-flash]");
    this.effects = new Effects(this.scene, this.textures, this.flash);
    this.hud = {
      progress: container.querySelector("[data-ride-progress]"),
      gauge: container.querySelector("[data-ride-gauge]"),
    };
    this.ctx = {
      scene: this.scene, car: this.car, audio: this.audio, effects: this.effects, scenery: this.scenery,
      materials: this.materials, textures: this.textures, emit: (event) => this._emit(event),
    };
    this.director = new Director(this.ctx);
    this.timer = new THREE.Timer();
    this.elapsed = 0;
    this._bindInput();
    this._resize();
    this.resizeObserver = new ResizeObserver(() => this._resize());
    this.resizeObserver.observe(container);
    this.renderer.setAnimationLoop(() => this._frame());
  }

  // ---- public API ------------------------------------------------------
  get drivePressed() {
    return this.keys.size > 0;
  }

  // Run one recorded ride. Resolves with how it ended.
  run({minutes, control = "keys", mindState = () => "STOP"}) {
    this.audio.start();
    this.control = control;
    this.mindState = mindState;
    this.duration = minutes * 60;
    this.rideTime = 0;
    this.sampleClock = 0;
    this.lastDecoder = null;
    this.car.reset(this.car.s);
    this.keys.clear();
    this.director.start();
    this.mode = "ride";
    this.container.classList.add("is-riding");
    this._emit({type: "ride_start", minutes, control, preset: this.environment.name});
    return new Promise((resolve) => { this._resolve = resolve; });
  }

  abort() {
    if (this.mode === "ride") this._end("aborted");
  }

  // ---- internals -------------------------------------------------------
  _emit(event) {
    if (this.mode !== "ride" && event.type !== "ride_end") return;
    this.onEvent({t: now(), ...event});
  }

  _end(reason) {
    if (this.keys.size) this._emit({type: "key", down: false, code: "end"});
    this.keys.clear();
    this._emit({type: "ride_end", reason});
    this.mode = "idle";
    this.director.stop();
    this.container.classList.remove("is-riding");
    this.hud.progress.style.transform = "scaleX(0)";
    const resolve = this._resolve;
    this._resolve = null;
    if (resolve) resolve(reason);
  }

  _bindInput() {
    const down = (event) => {
      if (this.mode !== "ride") return;
      if (event.code === "Escape") {
        event.preventDefault();
        this.abort();
        return;
      }
      if (!DRIVE_KEYS.has(event.code)) return;
      event.preventDefault();
      if (event.repeat || this.keys.has(event.code)) return;
      const wasIdle = this.keys.size === 0;
      this.keys.add(event.code);
      if (wasIdle) this._emit({type: "key", down: true, code: event.code});
    };
    const up = (event) => {
      if (!DRIVE_KEYS.has(event.code) || !this.keys.has(event.code)) return;
      event.preventDefault();
      this.keys.delete(event.code);
      if (this.keys.size === 0) this._emit({type: "key", down: false, code: event.code});
    };
    const release = () => {
      if (this.keys.size === 0) return;
      this.keys.clear();
      this._emit({type: "key", down: false, code: "blur"});
    };
    const hidden = () => {
      if (document.visibilityState === "hidden") release();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", release);
    document.addEventListener("visibilitychange", hidden);
    this.listeners = [["keydown", down, window], ["keyup", up, window], ["blur", release, window], ["visibilitychange", hidden, document]];
  }

  _resize() {
    const {clientWidth: width, clientHeight: height} = this.container;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  _drive() {
    if (this.mode !== "ride") return this.car.speed < IDLE_SPEED;
    if (this.control === "mind") {
      const state = this.mindState();
      if (state !== this.lastDecoder) {
        this.lastDecoder = state;
        this._emit({type: "decoder", state});
      }
      return state === "GO";
    }
    return this.drivePressed;
  }

  _frame() {
    this.timer.update();
    const dt = Math.min(0.05, this.timer.getDelta());
    if (this.paused) {
      this.renderer.render(this.scene, this.camera);
      return;
    }
    this.elapsed += dt;
    const drive = this._drive();
    if (this.mode === "ride") {
      this.rideTime += dt;
      const ending = this.rideTime > this.duration - 6;
      this.director.step(dt, drive, {allowSpawn: !ending});
      this.sampleClock += dt;
      if (this.sampleClock >= SPEED_SAMPLE_S) {
        this.sampleClock = 0;
        this._emit({type: "speed", v: Math.round(this.car.speed * 10) / 10});
      }
      this.hud.progress.style.transform = `scaleX(${Math.min(1, this.rideTime / this.duration).toFixed(4)})`;
      if ((this.rideTime >= this.duration && !this.director.busy) || this.rideTime > this.duration + 25) {
        this._end("complete");
      }
    } else {
      this.car.update(dt, drive);
    }
    this.road.update(this.car.s);
    this.scenery.update(this.car.s);
    this.environment.update(this.car.position, this.elapsed);
    this.effects.update(dt);
    this.audio.update(this.car.speed, drive);
    this._gauge();
    this.renderer.render(this.scene, this.camera);
  }

  _gauge() {
    const canvas = this.hud.gauge;
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    const size = canvas.clientWidth * ratio;
    if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
    const g = canvas.getContext("2d");
    g.clearRect(0, 0, size, size);
    const center = size / 2;
    const radius = size * 0.42;
    const start = Math.PI * 0.8;
    const sweep = Math.PI * 1.4;
    g.lineCap = "round";
    g.lineWidth = size * 0.035;
    g.strokeStyle = "rgba(255,255,255,0.16)";
    g.beginPath(); g.arc(center, center, radius, start, start + sweep); g.stroke();
    g.strokeStyle = "rgba(255,196,92,0.95)";
    g.beginPath(); g.arc(center, center, radius, start, start + sweep * (this.car.speed / MAX_SPEED) + 0.001); g.stroke();
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    for (const [name, handler, target] of this.listeners) target.removeEventListener(name, handler);
    this.audio.stop();
    this.renderer.dispose();
  }
}
