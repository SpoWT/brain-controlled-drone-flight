// Sound and visual effects.
import * as THREE from "../../vendor/three.module.min.js";

// Procedural car and event sound. No samples, no speech.
export class RideAudio {
  constructor() {
    this.context = null;
  }

  start() {
    if (this.context) {
      this.context.resume().catch(() => {});
      return;
    }
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) return;
    const ctx = new Context();
    this.context = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0.9;
    this.master.connect(ctx.destination);
    this.noise = this._noiseBuffer(3);

    // Engine: two detuned saws through a low-pass.
    this.engineFilter = ctx.createBiquadFilter();
    this.engineFilter.type = "lowpass";
    this.engineFilter.frequency.value = 420;
    this.engineGain = ctx.createGain();
    this.engineGain.gain.value = 0.0;
    this.engineFilter.connect(this.engineGain).connect(this.master);
    this.engine = [0, 7].map((detune) => {
      const osc = ctx.createOscillator();
      osc.type = "sawtooth";
      osc.detune.value = detune;
      osc.frequency.value = 38;
      osc.connect(this.engineFilter);
      osc.start();
      return osc;
    });

    // Tyre roar and wind: filtered looping noise.
    this.roadGain = this._loop("bandpass", 650, 0.7);
    this.windGain = this._loop("lowpass", 380, 0.5);
  }

  _noiseBuffer(seconds) {
    const ctx = this.context;
    const buffer = ctx.createBuffer(1, ctx.sampleRate * seconds, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    let brown = 0;
    for (let i = 0; i < data.length; i += 1) {
      const white = Math.random() * 2 - 1;
      brown = (brown + 0.02 * white) / 1.02;
      data[i] = white * 0.5 + brown * 3;
    }
    return buffer;
  }

  _loop(type, frequency, q) {
    const ctx = this.context;
    const source = ctx.createBufferSource();
    source.buffer = this.noise;
    source.loop = true;
    const filter = ctx.createBiquadFilter();
    filter.type = type;
    filter.frequency.value = frequency;
    filter.Q.value = q;
    const gain = ctx.createGain();
    gain.gain.value = 0;
    source.connect(filter).connect(gain).connect(this.master);
    source.start();
    return gain;
  }

  update(speed, throttle) {
    if (!this.context) return;
    const now = this.context.currentTime;
    const ratio = speed / 21;
    const gear = ratio < 0.35 ? ratio / 0.35 : ratio < 0.7 ? (ratio - 0.35) / 0.35 : (ratio - 0.7) / 0.3;
    const rpm = 34 + gear * 46 + ratio * 18;
    for (const osc of this.engine) osc.frequency.setTargetAtTime(rpm, now, 0.08);
    this.engineFilter.frequency.setTargetAtTime(260 + rpm * 6 * (throttle ? 1.4 : 0.8), now, 0.1);
    this.engineGain.gain.setTargetAtTime(0.035 + (throttle ? 0.05 : 0.015), now, 0.12);
    this.roadGain.gain.setTargetAtTime(0.09 * ratio, now, 0.2);
    this.windGain.gain.setTargetAtTime(0.12 * ratio * ratio, now, 0.3);
  }

  // Short filtered noise burst: thuds, cracks, rumbles.
  burst({type = "lowpass", frequency = 200, q = 0.8, duration = 0.6, gain = 0.8, delay = 0, attack = 0.005}) {
    if (!this.context) return;
    const ctx = this.context;
    const start = ctx.currentTime + delay;
    const source = ctx.createBufferSource();
    source.buffer = this.noise;
    source.playbackRate.value = 0.8 + Math.random() * 0.4;
    const filter = ctx.createBiquadFilter();
    filter.type = type;
    filter.frequency.value = frequency;
    filter.Q.value = q;
    const envelope = ctx.createGain();
    envelope.gain.setValueAtTime(0.0001, start);
    envelope.gain.exponentialRampToValueAtTime(gain, start + attack);
    envelope.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    source.connect(filter).connect(envelope).connect(this.master);
    source.start(start, Math.random() * 2);
    source.stop(start + duration + 0.05);
  }

  tone({frequency, to = frequency, duration = 0.4, gain = 0.1, type = "sine", delay = 0}) {
    if (!this.context) return;
    const ctx = this.context;
    const start = ctx.currentTime + delay;
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(frequency, start);
    osc.frequency.exponentialRampToValueAtTime(Math.max(1, to), start + duration);
    const envelope = ctx.createGain();
    envelope.gain.setValueAtTime(0.0001, start);
    envelope.gain.exponentialRampToValueAtTime(gain, start + 0.01);
    envelope.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    osc.connect(envelope).connect(this.master);
    osc.start(start);
    osc.stop(start + duration + 0.05);
  }

  thud(strength = 1) {
    this.burst({frequency: 140, duration: 0.7, gain: 0.9 * strength});
    this.tone({frequency: 70, to: 38, duration: 0.5, gain: 0.35 * strength});
  }

  crack() {
    this.burst({type: "highpass", frequency: 1800, duration: 0.18, gain: 0.7});
    this.burst({type: "bandpass", frequency: 500, q: 1.5, duration: 0.35, gain: 0.5, delay: 0.03});
  }

  creak() {
    this.tone({frequency: 95, to: 60, duration: 0.9, gain: 0.08, type: "sawtooth"});
  }

  rumble(duration = 2.4) {
    this.burst({frequency: 110, q: 0.6, duration, gain: 0.85, attack: 0.15});
    for (let i = 0; i < 6; i += 1) this.burst({frequency: 260, duration: 0.25, gain: 0.35, delay: 0.2 + Math.random() * duration * 0.8});
  }

  thunder() {
    this.burst({type: "highpass", frequency: 2500, duration: 0.12, gain: 0.9});
    this.burst({frequency: 90, q: 0.5, duration: 3.2, gain: 1.0, attack: 0.04, delay: 0.05});
    this.burst({frequency: 160, duration: 1.4, gain: 0.5, delay: 0.5});
  }

  screech() {
    this.tone({frequency: 1250, to: 980, duration: 0.7, gain: 0.03, type: "triangle"});
  }

  crash() {
    this.thud(1.2);
    this.burst({type: "highpass", frequency: 3200, duration: 0.5, gain: 0.35, delay: 0.02});
  }

  stop() {
    if (this.context) this.context.suspend().catch(() => {});
  }
}

// Dust clouds, sparks, lightning bolts, and full-screen flashes.

const MAX_PARTICLES = 900;

export class Effects {
  constructor(scene, textures, flashElement) {
    this.scene = scene;
    this.flashElement = flashElement;
    this.flashLevel = 0;
    this.flashDecay = 1;
    const geometry = new THREE.BufferGeometry();
    this.positions = new Float32Array(MAX_PARTICLES * 3);
    this.colors = new Float32Array(MAX_PARTICLES * 4); // rgba: alpha fades out
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(this.colors, 4));
    this.particles = [];
    this.points = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 2.6, map: textures.dust, vertexColors: true, transparent: true, depthWrite: false, sizeAttenuation: true,
    }));
    this.points.frustumCulled = false;
    scene.add(this.points);
    this.bolts = [];
  }

  // A billowing cloud of dust or smoke at a world position.
  dust(origin, {count = 60, spread = 3, rise = 1.2, color = 0xb8a88c, life = 3.5} = {}) {
    const tint = new THREE.Color(color);
    for (let i = 0; i < count && this.particles.length < MAX_PARTICLES; i += 1) {
      this.particles.push({
        x: origin.x + (Math.random() - 0.5) * spread, y: origin.y + Math.random() * 0.6,
        z: origin.z + (Math.random() - 0.5) * spread,
        vx: (Math.random() - 0.5) * spread * 0.9, vy: Math.random() * rise, vz: (Math.random() - 0.5) * spread * 0.9,
        age: 0, life: life * (0.6 + Math.random() * 0.6), color: tint,
      });
    }
  }

  // A short-lived jagged lightning bolt from the clouds to `target`.
  bolt(target) {
    const points = [];
    let x = target.x + (Math.random() - 0.5) * 60;
    let z = target.z - 40;
    for (let y = 260; y > target.y; y -= 12) {
      points.push(new THREE.Vector3(x, y, z));
      const t = 1 - y / 260;
      x += (target.x - x) * t * 0.35 + (Math.random() - 0.5) * 14;
      z += (target.z - z) * t * 0.35 + (Math.random() - 0.5) * 10;
    }
    points.push(target.clone());
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({color: 0xf4f2ff, transparent: true, fog: false}),
    );
    this.scene.add(line);
    this.bolts.push({line, age: 0});
  }

  flash(color = "#ffffff", strength = 1, decay = 2.5) {
    this.flashElement.style.background = color;
    this.flashLevel = Math.max(this.flashLevel, strength);
    this.flashDecay = decay;
  }

  update(dt) {
    let alive = 0;
    for (const p of this.particles) {
      p.age += dt;
      if (p.age >= p.life) continue;
      p.vx *= 1 - dt * 0.9;
      p.vz *= 1 - dt * 0.9;
      p.vy = p.vy * (1 - dt * 0.6) + dt * 0.05;
      p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
      const fade = 1 - p.age / p.life;
      this.positions.set([p.x, p.y, p.z], alive * 3);
      this.colors.set([p.color.r, p.color.g, p.color.b, fade * 0.6], alive * 4);
      this.particles[alive] = p;
      alive += 1;
    }
    this.particles.length = alive;
    const geometry = this.points.geometry;
    geometry.setDrawRange(0, alive);
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;

    this.bolts = this.bolts.filter((bolt) => {
      bolt.age += dt;
      bolt.line.material.opacity = bolt.age < 0.28 ? (Math.sin(bolt.age * 90) > -0.3 ? 1 : 0.15) : 0;
      if (bolt.age > 0.35) {
        this.scene.remove(bolt.line);
        bolt.line.geometry.dispose();
        bolt.line.material.dispose();
        return false;
      }
      return true;
    });

    this.flashLevel = Math.max(0, this.flashLevel - dt * this.flashDecay);
    this.flashElement.style.opacity = this.flashLevel.toFixed(3);
  }
}
