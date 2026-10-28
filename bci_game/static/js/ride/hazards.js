// Every hazard: the shared lifecycle, then one class per hazard kind.
import * as THREE from "../../vendor/three.module.min.js";
import {glowSprite, makeCar, makeDeer, makeSignal, makeTree, rockGeometry} from "./assets.js";
import {heading, LANE, place, ROAD_HALF, SHOULDER} from "./road.js";

// Shared hazard lifecycle: appear -> demand_stop -> clear -> end.

const GRAVITY = 9.81;
let counter = 0;

export const rand = (lo, hi) => lo + Math.random() * (hi - lo);
export const easeIn = (t) => t * t;
export const easeOut = (t) => 1 - (1 - t) * (1 - t);

export class Hazard {
  constructor(ctx, kind) {
    counter += 1;
    this.ctx = ctx;
    this.kind = kind;
    this.id = `${kind}-${counter}`;
    this.group = new THREE.Group();
    ctx.scene.add(this.group);
    this.age = 0;
    this.demanded = false;
    this.cleared = false;
    this.done = false;
    this.stillFor = 0;
    this.emit("appear");
  }

  emit(phase, extra = {}) {
    this.ctx.emit({type: "hazard", kind: this.kind, phase, id: this.id, ...extra});
  }

  demandStop() {
    if (this.demanded) return;
    this.demanded = true;
    this.emit("demand_stop", {distance: Math.round(this.distance())});
  }

  clear() {
    if (this.cleared) return;
    this.cleared = true;
    this.emit("clear");
  }

  finish() {
    if (this.done) return;
    this.done = true;
    this.emit("end");
  }

  // Distance from the car's front bumper to this hazard's anchor point.
  distance() {
    return this.anchorS - this.ctx.car.front;
  }

  // The road position the car's front must not pass, or null when passable.
  blockS() {
    return null;
  }

  onCollision() {}

  // Track how long the rider has been fully stopped since the stop demand.
  trackStill(dt) {
    if (this.demanded && !this.cleared && this.ctx.car.speed < 0.2) this.stillFor += dt;
  }

  tick(dt) {
    this.age += dt;
    this.trackStill(dt);
    this.update(dt);
  }

  update() {}

  dispose() {
    this.ctx.scene.remove(this.group);
    this.group.traverse((child) => {
      if (child.isMesh && child.geometry && !child.geometry.userData.shared) child.geometry.dispose();
    });
  }
}

// An intersection with an overhead signal. `red_light` turns amber then red as
// you approach and later goes green; `green_light` stays green (a no-stop catch).

const AMBER_S = 1.3;

function flat(width, depth, material) {
  const geometry = new THREE.PlaneGeometry(width, depth);
  geometry.rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.receiveShadow = true;
  return mesh;
}

export class TrafficLight extends Hazard {
  constructor(ctx, kind = "red_light") {
    super(ctx, kind);
    this.anchorS = ctx.car.front + 100;
    this.lineS = this.anchorS;
    this.crossS = this.lineS + 9;
    this.trigger = rand(58, 72);
    this.hold = rand(1.5, 3);
    this.state = "green";
    this.stateAge = 0;
    this.crossers = [];
    ctx.scenery.reserve(this.lineS - 2, this.crossS + 8);
    this._build();
    this._setLamps("green");
  }

  _build() {
    const {ctx} = this;
    const yaw = heading(this.lineS);
    const paint = ctx.materials.paintLine;
    const stop = flat(3.5, 0.45, paint);
    stop.position.copy(place(this.lineS, LANE, 0.04));
    stop.rotation.y = yaw;
    this.group.add(stop);
    for (let k = 0; k < 8; k += 1) {
      const stripe = flat(0.5, 3.0, paint);
      stripe.position.copy(place(this.lineS + 3.2, -3.15 + k * 0.9, 0.04));
      stripe.rotation.y = yaw;
      this.group.add(stripe);
    }
    const cross = flat(420, 8.5, ctx.materials.crossRoad);
    cross.position.copy(place(this.crossS, 0, 0.025));
    cross.rotation.y = yaw;
    this.group.add(cross);
    const {group, lamps} = makeSignal(ctx.textures.glow);
    group.position.copy(place(this.lineS + 1.5, ROAD_HALF + SHOULDER * 0.6, 0));
    group.rotation.y = yaw;
    this.group.add(group);
    this.lamps = lamps;
  }

  _setLamps(state) {
    for (const [name, list] of Object.entries(this.lamps)) {
      const on = name === state;
      for (const {material, halo} of list) {
        material.emissiveIntensity = on ? 3.4 : 0.04;
        halo.material.opacity = on ? 0.95 : 0;
      }
    }
  }

  _enter(state) {
    this.state = state;
    this.stateAge = 0;
    this._setLamps(state);
  }

  _spawnCrosser(direction) {
    const car = makeCar(this.ctx.textures.glow, [0x2d4f7c, 0xc9c7c0, 0x1d1f22, 0x6d6f2c][Math.floor(Math.random() * 4)]);
    this.group.add(car.group);
    this.crossers.push({car, l: -direction * 140, direction, speed: rand(11, 15)});
  }

  blockS() {
    const passed = this.ctx.car.front > this.lineS + 0.5;
    return this.state === "red" && !passed && !this.redStartedPast ? this.lineS : null;
  }

  update(dt) {
    this.stateAge += dt;
    const distance = this.distance();
    if (this.kind === "green_light" && !this.cleared && distance <= this.trigger) this.clear();
    if (this.kind === "red_light") {
      if (this.state === "green" && !this.demanded && distance <= this.trigger) {
        this._enter("amber");
        this.demandStop();
      } else if (this.state === "amber" && this.stateAge >= AMBER_S) {
        this.redStartedPast = this.ctx.car.front > this.lineS;
        this._enter("red");
        this._spawnCrosser(1);
        this.nextCrosser = rand(1.5, 3);
      } else if (this.state === "red") {
        if (this.nextCrosser !== null && this.stateAge > this.nextCrosser) {
          this._spawnCrosser(-1);
          this.nextCrosser = null;
        }
        if (this.stillFor >= this.hold || this.ctx.car.front > this.lineS + 1) {
          this._enter("green");
          this.clear();
        }
      }
    }
    for (const crosser of this.crossers) {
      crosser.l += crosser.direction * crosser.speed * dt;
      const lane = crosser.direction > 0 ? 2.1 : -2.1;
      crosser.car.group.position.copy(place(this.crossS + lane, crosser.l, 0));
      crosser.car.group.rotation.y = heading(this.crossS) - crosser.direction * Math.PI / 2;
    }
    if (this.ctx.car.front > this.crossS + 40) {
      this.ctx.scenery.release(this.lineS - 2, this.crossS + 8);
      this.finish();
    }
  }
}

// A tall roadside spruce snaps and falls across both lanes, then is dragged clear.

const TREE_HEIGHT = 20;
const REST_ANGLE = 1.5;

export class FallenTree extends Hazard {
  constructor(ctx) {
    super(ctx, "fallen_tree");
    this.anchorS = ctx.car.front + 92;
    this.side = Math.random() < 0.5 ? -1 : 1;
    this.baseL = this.side * (ROAD_HALF + SHOULDER + 2.2);
    this.trigger = rand(66, 80); // falls well ahead: ~1.7-2.4 s to react at full speed
    this.hold = rand(1.2, 2.2);
    this.angle = 0;
    this.spin = 0;
    this.state = "standing";
    this.slide = 0;
    ctx.scenery.reserve(this.anchorS - 7, this.anchorS + 7);
    this.yaw = heading(this.anchorS);
    this.pivot = new THREE.Group();
    this.pivot.position.copy(place(this.anchorS, this.baseL, 0));
    this.pivot.rotation.y = this.yaw;
    this.tilt = new THREE.Group();
    this.tilt.add(makeTree(ctx.materials, TREE_HEIGHT, 5));
    this.pivot.add(this.tilt);
    this.group.add(this.pivot);
    this.stump = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.5, 0.9, 9), ctx.materials.bark);
    this.stump.position.copy(place(this.anchorS, this.baseL, 0.45));
    this.stump.visible = false;
    this.group.add(this.stump);
  }

  blockS() {
    return this.angle > 0.7 && this.state !== "cleared" ? this.anchorS - 3.0 : null;
  }

  _impact() {
    const {ctx} = this;
    ctx.audio.thud(1.0);
    ctx.car.impulse(0.45);
    for (let k = 0; k <= 4; k += 1) {
      ctx.effects.dust(place(this.anchorS, this.baseL - this.side * k * 3.4, 0.4), {count: 26, spread: 3, color: 0xa69a82});
    }
  }

  update(dt) {
    const {ctx} = this;
    let shiver = 0;
    if (this.state === "standing") {
      if (this.distance() <= this.trigger + 12) shiver = Math.sin(this.age * 22) * 0.006;
      if (this.distance() <= this.trigger) {
        this.state = "falling";
        this.angle = 0.03;
        this.stump.visible = true;
        ctx.audio.crack();
        ctx.audio.creak();
        this.demandStop();
      }
    } else if (this.state === "falling") {
      this.spin += ((3 * GRAVITY) / (2 * TREE_HEIGHT)) * Math.sin(this.angle) * dt;
      this.angle += this.spin * dt;
      if (this.angle >= REST_ANGLE) {
        this.angle = REST_ANGLE;
        if (!this.landed) {
          this.landed = true;
          this._impact();
        }
        this.spin = -this.spin * 0.2;
        if (Math.abs(this.spin) < 0.08) {
          this.spin = 0;
          this.state = "down";
        }
      }
    } else if (this.state === "down") {
      if (this.stillFor >= this.hold) {
        this.state = "dragging";
        ctx.audio.rumble(1.6);
        ctx.effects.dust(place(this.anchorS, 0, 0.3), {count: 40, spread: 6, color: 0xa69a82});
      }
    } else if (this.state === "dragging") {
      this.slide = Math.min(1, this.slide + dt / 1.0);
      const shift = easeOut(this.slide) * 22;
      this.pivot.position.copy(place(this.anchorS + this.slide * 3, this.baseL + this.side * shift, 0));
      this.pivot.rotation.y = this.yaw + this.side * this.slide * 0.25;
      if (this.slide >= 1) {
        this.state = "cleared";
        this.clear();
      }
    }
    this.tilt.rotation.z = this.side * (this.angle + shiver);
    if (ctx.car.front > this.anchorS + 10) {
      ctx.scenery.release(this.anchorS - 7, this.anchorS + 7);
      this.finish();
    }
  }
}

// Boulders break off a roadside outcrop, bounce into the lanes, and later
// tumble on down-slope off the far edge of the road.


export class Rockfall extends Hazard {
  constructor(ctx) {
    super(ctx, "rockfall");
    this.anchorS = ctx.car.front + 96;
    this.side = Math.random() < 0.5 ? -1 : 1;
    this.trigger = rand(70, 84); // rocks land well ahead: ~1.8-2.5 s to react at full speed
    this.hold = rand(1.2, 2.2);
    this.state = "waiting";
    this.rollingOff = false;
    ctx.scenery.reserve(this.anchorS - 22, this.anchorS + 22);
    this._outcrop();
    this.rocks = [];
  }

  _outcrop() {
    const {ctx} = this;
    const yaw = heading(this.anchorS);
    for (let k = 0; k < 9; k += 1) {
      const rock = new THREE.Mesh(rockGeometry(100 + k, 1), ctx.materials.rock);
      const size = rand(4, 8);
      rock.scale.set(size * rand(0.8, 1.4), size * rand(1.0, 1.8), size);
      rock.position.copy(place(this.anchorS + rand(-18, 18), this.side * (ROAD_HALF + SHOULDER + 5 + rand(0, 8)), size * 0.6));
      rock.rotation.set(rand(0, 3), yaw + rand(-0.5, 0.5), rand(0, 3));
      rock.castShadow = true;
      rock.receiveShadow = true;
      this.group.add(rock);
    }
  }

  _release() {
    const {ctx} = this;
    const count = 7;
    for (let k = 0; k < count; k += 1) {
      const radius = rand(0.45, 1.05);
      const mesh = new THREE.Mesh(rockGeometry(200 + k, 0), ctx.materials.rock);
      mesh.scale.setScalar(radius);
      mesh.castShadow = true;
      this.group.add(mesh);
      // Most land in our (right-hand) lane so the way is clearly blocked.
      const target = k < 4 ? rand(0.4, 3.2) : rand(-3.2, 3.2);
      this.rocks.push({
        mesh, radius, delay: k * 0.18 + rand(0, 0.3),
        s: this.anchorS + rand(-4, 4), l0: this.side * (ROAD_HALF + SHOULDER + 6), l: 0, target,
        y: rand(9, 14), vy: 0, duration: rand(1.6, 2.3), t: 0, spin: rand(2, 5),
        rollFrom: 0, rollTo: 0, rollT: 0,
      });
    }
    ctx.audio.rumble(2.6);
    ctx.effects.dust(place(this.anchorS, this.side * (ROAD_HALF + SHOULDER + 6), 10), {count: 50, spread: 6, color: 0x9d968a});
  }

  blockS() {
    if (this.state !== "falling" && this.state !== "resting") return null;
    let nearest = null;
    for (const rock of this.rocks) {
      if (rock.t > rock.delay && Math.abs(rock.l) < ROAD_HALF + 0.4 && rock.y < 3) {
        const front = rock.s - rock.radius - 0.4;
        nearest = nearest === null ? front : Math.min(nearest, front);
      }
    }
    return nearest;
  }

  _step(rock, dt) {
    rock.t += dt;
    if (rock.t < rock.delay) {
      rock.mesh.visible = false;
      return;
    }
    rock.mesh.visible = true;
    const local = rock.t - rock.delay;
    if (this.rollingOff) {
      rock.rollT = Math.min(1, rock.rollT + dt / 1.3);
      rock.l = rock.rollFrom + (rock.rollTo - rock.rollFrom) * easeOut(rock.rollT);
      rock.y = rock.radius * 0.8 - Math.max(0, Math.abs(rock.l) - (ROAD_HALF + SHOULDER)) * 0.35;
    } else {
      rock.l = rock.l0 + (rock.target - rock.l0) * easeOut(Math.min(1, local / rock.duration));
      rock.vy -= GRAVITY * dt;
      rock.y += rock.vy * dt;
      const floor = rock.radius * 0.8;
      if (rock.y < floor) {
        rock.y = floor;
        if (Math.abs(rock.vy) > 2.5) {
          this.ctx.audio.thud(Math.min(0.5, Math.abs(rock.vy) / 20));
          this.ctx.effects.dust(place(rock.s, rock.l, 0.2), {count: 6, spread: 1.4, color: 0xa69a82});
        }
        rock.vy = Math.abs(rock.vy) > 1.2 ? -rock.vy * 0.38 : 0;
      }
    }
    const moving = this.rollingOff ? 1 - rock.rollT : Math.max(0, 1 - local / rock.duration);
    rock.mesh.rotation.x += rock.spin * moving * dt;
    rock.mesh.rotation.z -= this.side * rock.spin * moving * dt;
    rock.mesh.position.copy(place(rock.s, rock.l, rock.y));
  }

  update(dt) {
    const {ctx} = this;
    if (this.state === "waiting" && this.distance() <= this.trigger) {
      this.state = "falling";
      this._release();
      this.demandStop();
    }
    for (const rock of this.rocks) this._step(rock, dt);
    if (this.state === "falling" && this.rocks.every((rock) => rock.t > rock.delay + rock.duration)) {
      this.state = "resting";
    }
    if (this.state === "resting" && this.stillFor >= this.hold) {
      this.state = "rolling_off";
      this.rollingOff = true;
      ctx.audio.rumble(2.0);
      for (const rock of this.rocks) {
        rock.rollFrom = rock.l;
        rock.rollTo = -this.side * (ROAD_HALF + SHOULDER + rand(4, 9));
        rock.rollT = 0;
      }
    }
    if (this.state === "rolling_off" && this.rocks.every((rock) => rock.rollT >= 0.75)) {
      this.state = "cleared";
      this.clear();
    }
    if (ctx.car.front > this.anchorS + 12) {
      ctx.scenery.release(this.anchorS - 22, this.anchorS + 22);
      this.finish();
    }
  }
}

// A sudden shock with nothing to read: a lightning strike just off the road,
// or a hard jolt as the car slams through a hidden pothole.

export class Shock extends Hazard {
  constructor(ctx) {
    super(ctx, "shock");
    this.variant = Math.random() < 0.6 ? "lightning" : "jolt";
    this.anchorS = ctx.car.front + 50;
    this.delay = rand(0.4, 1.6);
    this.hold = rand(2, 3.5);
    this.fired = false;
    this.fire = null;
  }

  _strike() {
    const {ctx} = this;
    if (this.variant === "lightning") {
      const side = Math.random() < 0.5 ? -1 : 1;
      this.anchorS = ctx.car.front + rand(40, 65);
      const target = place(this.anchorS, side * (ROAD_HALF + SHOULDER + rand(3, 9)), 0);
      ctx.effects.bolt(target);
      ctx.effects.flash("#eef0ff", 0.95, 1.4);
      window.setTimeout(() => ctx.effects.bolt(target), 140);
      ctx.audio.thunder();
      ctx.car.impulse(0.7);
      const light = new THREE.PointLight(0xff8a3a, 18, 30, 1.6);
      light.position.copy(target).setY(1.2);
      const glow = glowSprite(ctx.textures.glow, 0xff7a2a, 4);
      glow.position.copy(light.position);
      this.group.add(light, glow);
      this.fire = {light, glow, origin: target};
    } else {
      this.anchorS = ctx.car.front;
      ctx.car.impulse(2.0);
      ctx.audio.crash();
      ctx.effects.flash("#ffffff", 0.18, 3);
      ctx.effects.dust(place(ctx.car.s + 3, 1.75, 0.2), {count: 18, spread: 2, color: 0x8f877a, life: 1.6});
    }
    this.demandStop();
  }

  update(dt) {
    if (!this.fired && this.age >= this.delay) {
      this.fired = true;
      this._strike();
    }
    if (this.fire) {
      const flicker = 0.7 + Math.sin(this.age * 23) * 0.15 + Math.sin(this.age * 37) * 0.15;
      const fade = Math.max(0, 1 - (this.age - this.delay) / 7);
      this.fire.light.intensity = 18 * flicker * fade;
      this.fire.glow.material.opacity = flicker * fade;
      if (Math.random() < dt * 10 * fade) {
        this.ctx.effects.dust(this.fire.origin, {count: 2, spread: 1, rise: 2.5, color: 0x5b5652, life: 4});
      }
    }
    const since = this.age - this.delay;
    if (this.fired && !this.cleared && (this.stillFor >= this.hold || since > 6)) this.clear();
    if (this.cleared && since > 8) this.finish();
  }
}

// A deer bolts out of the trees, freezes in the lane, then leaps away.

const DEER_SPEED = 8.5;

export class Deer extends Hazard {
  constructor(ctx) {
    super(ctx, "deer");
    this.anchorS = ctx.car.front + 120;
    this.side = Math.random() < 0.5 ? -1 : 1;
    this.dir = -this.side;
    this.l = this.side * (ROAD_HALF + SHOULDER + 7);
    this.stand = rand(0.6, 2.6) * (Math.random() < 0.75 ? 1 : -1);
    this.trigger = rand(42, 56);
    this.hold = rand(2, 4);
    this.state = "hidden";
    this.stride = 0;
    const {group, legs, head} = makeDeer();
    this.deer = group;
    this.legs = legs;
    this.head = head;
    this.deer.scale.setScalar(1.05);
    this.group.add(group);
    this._pose(0);
  }

  blockS() {
    return Math.abs(this.l) < ROAD_HALF + 0.6 && this.state !== "gone" ? this.anchorS - 1.1 : null;
  }

  onCollision() {
    this.state = "leaving";
  }

  _pose(bob) {
    this.deer.position.copy(place(this.anchorS, this.l, bob));
    this.deer.rotation.y = heading(this.anchorS) + (this.dir > 0 ? 0 : Math.PI);
  }

  _gallop(dt) {
    this.stride += dt * 11;
    const swing = Math.sin(this.stride);
    this.legs[0].rotation.z = swing * 0.9;
    this.legs[1].rotation.z = swing * 0.7;
    this.legs[2].rotation.z = -swing * 0.9;
    this.legs[3].rotation.z = -swing * 0.7;
    this.deer.rotation.z = Math.cos(this.stride) * 0.08;
    return Math.max(0, Math.sin(this.stride)) * 0.45;
  }

  update(dt) {
    let bob = 0;
    if (this.state === "hidden" && this.distance() <= this.trigger) {
      this.state = "running";
      this.demandStop();
    }
    if (this.state === "running" || this.state === "leaving") {
      bob = this._gallop(dt);
      this.l += this.dir * DEER_SPEED * (this.state === "leaving" ? 1.25 : 1) * dt;
      this.head.rotation.y = 0;
      if (this.state === "running" && (this.l - this.stand) * this.dir >= 0) {
        this.l = this.stand;
        this.state = "frozen";
      }
      if (this.state === "leaving" && Math.abs(this.l) > ROAD_HALF + SHOULDER && Math.sign(this.l) === this.dir) {
        if (!this.cleared) this.clear();
        if (Math.abs(this.l) > 30) this.state = "gone";
      }
    } else if (this.state === "frozen") {
      for (const leg of this.legs) leg.rotation.z *= 0.8;
      this.deer.rotation.z = 0;
      this.head.rotation.y = (this.dir > 0 ? -1 : 1) * Math.min(1, this.age) * 0.9;
      if (this.stillFor >= this.hold) this.state = "leaving";
    }
    this._pose(bob);
    if (this.ctx.car.front > this.anchorS + 30) this.finish();
  }
}

// A car pulls in ahead, cruises, then brakes hard to a stop; later it drives off.

const CRUISE_GAP = 34;
const BRAKE = 6.5;
const PULL_AWAY = 3.8;
const LEAVE_SPEED = 28;   // clearly faster than our 21 m/s top speed, so it pulls away
const SAFE_GAP = 45;      // once this far ahead and leaving, it no longer blocks us

export class LeadCar extends Hazard {
  constructor(ctx) {
    super(ctx, "lead_car");
    this.carS = ctx.car.front + 55;
    this.anchorS = this.carS;
    this.speed = Math.min(15, Math.max(ctx.car.speed * 0.8, 10));
    this.state = "cruising";
    this.cruiseFor = rand(1.5, 3);
    this.hold = rand(1.2, 2.2);
    const colors = [0x8a1c1c, 0x23364f, 0xb8b6ae, 0x2b2d30, 0x4a5a3a];
    this.model = makeCar(ctx.textures.glow, colors[Math.floor(Math.random() * colors.length)]);
    this.group.add(this.model.group);
    this._brakeLights(false);
  }

  _brakeLights(on) {
    this.model.lampMaterial.emissiveIntensity = on ? 4.5 : 0.35;
    for (const glow of this.model.glows) glow.material.opacity = on ? 0.95 : 0.0;
  }

  // While it cruises we follow it at its pace instead of running into it.
  speedCap() {
    if (this.state !== "cruising") return null;
    return this.carS - this.ctx.car.front < CRUISE_GAP + 4 ? this.speed : null;
  }

  blockS() {
    if (this.state === "gone") return null;
    if (this.state === "leaving" && this.carS - this.ctx.car.front > SAFE_GAP) return null;
    return this.carS - 2.2 - 1.2;
  }

  update(dt) {
    const {ctx} = this;
    const gap = this.carS - ctx.car.front;
    if (this.state === "cruising") {
      // Slightly slower than our top speed, so we always close in behind it.
      const target = ctx.car.speed + (gap - CRUISE_GAP) * 0.35;
      this.speed += (Math.min(16.5, Math.max(6, target)) - this.speed) * Math.min(1, dt * 1.5);
      this.cruiseFor -= dt;
      if (this.cruiseFor <= 0 && (gap < CRUISE_GAP + 12 || this.cruiseFor < -6)) {
        this.state = "braking";
        this._brakeLights(true);
        ctx.audio.screech();
        this.demandStop();
      }
    } else if (this.state === "braking") {
      this.speed = Math.max(0, this.speed - BRAKE * dt);
      if (this.speed === 0) this.state = "stopped";
    } else if (this.state === "stopped") {
      if (this.stillFor >= this.hold) {
        this.state = "leaving";
        this._brakeLights(false);
        this.clear();
      }
    } else if (this.state === "leaving") {
      this.speed = Math.min(LEAVE_SPEED, this.speed + PULL_AWAY * dt);
      if (gap > 160) {
        this.state = "gone";
        this.finish();
      }
    }
    this.carS += this.speed * dt;
    this.anchorS = this.carS;
    this.model.group.position.copy(place(this.carS, LANE, 0));
    this.model.group.rotation.y = heading(this.carS);
    // Brake dive.
    this.model.group.rotation.x = this.state === "braking" ? -0.02 : 0;
  }
}
