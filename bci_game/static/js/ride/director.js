// Schedules one hazard at a time from a shuffled, balanced deck and enforces
// that obstacles are solid: driving into one is a collision.
import {Deer, FallenTree, LeadCar, Rockfall, Shock, TrafficLight, rand} from "./hazards.js";

const DECK = [
  // Trees and rocks dominate: 12 of every 18 hazards.
  "fallen_tree", "fallen_tree", "fallen_tree", "fallen_tree", "fallen_tree", "fallen_tree",
  "rockfall", "rockfall", "rockfall", "rockfall", "rockfall", "rockfall",
  "red_light", "red_light", "shock", "deer", "lead_car", "green_light",
];
const MIN_SPAWN_SPEED = 5;
const RETIRE_SPEED = 5;        // once rolling again after a clear, the next hazard may start
const EARLY_RETIRE = new Set(["fallen_tree", "rockfall", "shock", "deer", "red_light", "green_light", "lead_car"]);
const CAR_LENGTH_AHEAD = 2.3;
const MAX_HAZARD_S = 75; // failsafe: no hazard may hold the road longer than this

function shuffled(items) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function spawn(ctx, kind) {
  switch (kind) {
    case "red_light": return new TrafficLight(ctx, "red_light");
    case "green_light": return new TrafficLight(ctx, "green_light");
    case "fallen_tree": return new FallenTree(ctx);
    case "rockfall": return new Rockfall(ctx);
    case "shock": return new Shock(ctx);
    case "deer": return new Deer(ctx);
    case "lead_car": return new LeadCar(ctx);
    default: throw new Error(`unknown hazard ${kind}`);
  }
}

export class Director {
  constructor(ctx) {
    this.ctx = ctx;
    this.active = null;
    this.retiring = []; // cleared hazards still animating behind the next one
    this.deck = [];
    this.enabled = false;
    this.gap = 0;
  }

  start() {
    this.clearActive();
    this.deck = shuffled(DECK);
    this.enabled = true;
    this.gap = rand(2, 3.5);
  }

  // Make `kind` the very next hazard (used by the ?debug hook).
  queueNext(kind) {
    this.deck.push(kind);
    this.gap = Math.min(this.gap, 0.5);
  }

  stop() {
    this.enabled = false;
    this.clearActive();
  }

  clearActive() {
    if (this.active) this.active.dispose();
    this.active = null;
    for (const hazard of this.retiring) hazard.dispose();
    this.retiring = [];
  }

  // A cleared, passable hazard hands the road to the next one while it finishes animating.
  _retireIfDone(hazard) {
    const passable = hazard.cleared && hazard.blockS() === null && EARLY_RETIRE.has(hazard.kind);
    if (!passable || this.ctx.car.speed < RETIRE_SPEED) return false;
    this.retiring.push(hazard);
    this.active = null;
    this.gap = rand(0.8, 1.6);
    return true;
  }

  _tickRetiring(dt) {
    this.retiring = this.retiring.filter((hazard) => {
      hazard.tick(dt);
      if (hazard.done || hazard.age > MAX_HAZARD_S) {
        hazard.dispose();
        return false;
      }
      return true;
    });
  }

  get busy() {
    return this.active !== null;
  }

  // Advance the car with obstacles treated as solid, then the hazard itself.
  step(dt, drive, {allowSpawn = true} = {}) {
    const {car} = this.ctx;
    this._tickRetiring(dt);
    const hazard = this.active;
    const block = hazard ? hazard.blockS() : null;
    const cap = block === null ? Infinity : block - CAR_LENGTH_AHEAD;
    const speedBefore = car.speed;
    car.update(dt, drive, cap);
    const speedCap = hazard?.speedCap ? hazard.speedCap() : null;
    if (speedCap !== null && car.speed > speedCap) car.speed = speedCap;
    if (hazard && block !== null && car.s >= cap - 1e-6 && speedBefore > 0.6 && !hazard.collided) {
      hazard.collided = true;
      car.crash(block);
      this.ctx.audio.crash();
      this.ctx.effects.flash("#ff3b2f", 0.5, 1.4);
      this.ctx.emit({type: "collision", hazard_id: hazard.id, kind: hazard.kind, speed: Math.round(speedBefore * 10) / 10});
      hazard.onCollision();
    }
    if (hazard) {
      hazard.tick(dt);
      if (hazard.age > MAX_HAZARD_S && !hazard.done) {
        if (!hazard.cleared) hazard.clear();
        hazard.emit("timeout");
        hazard.finish();
      }
      if (hazard.done) {
        hazard.dispose();
        this.active = null;
        this.gap = rand(0.8, 1.6); // short breather, then the next hazard
      } else {
        this._retireIfDone(hazard);
      }
    } else if (this.enabled && allowSpawn && car.speed > MIN_SPAWN_SPEED) {
      this.gap -= dt;
      if (this.gap <= 0) {
        if (this.deck.length === 0) this.deck = shuffled(DECK);
        this.active = spawn(this.ctx, this.deck.pop());
      }
    }
  }
}
