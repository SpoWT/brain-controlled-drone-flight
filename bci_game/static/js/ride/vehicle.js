// The rider's car: longitudinal physics, a first-person camera that dives
// under braking and shakes on impacts, and a cockpit frame around the view.
import * as THREE from "../../vendor/three.module.min.js";
import {LANE, heading, place} from "./road.js";

export const MAX_SPEED = 21;         // m/s (~75 km/h)
const ACCEL = 3.4;
const BRAKE = 7.2;
const EYE_HEIGHT = 1.22;

function cockpit(materials) {
  const group = new THREE.Group();
  // Wide crowned hood panel; its front edge droops toward the fenders.
  const hoodGeometry = new THREE.PlaneGeometry(4.2, 3.2, 48, 12);
  hoodGeometry.rotateX(-Math.PI / 2);
  const position = hoodGeometry.attributes.position;
  for (let i = 0; i < position.count; i += 1) {
    const x = position.getX(i);
    const z = position.getZ(i);
    position.setY(i, -0.05 * x * x - 0.0175 * (z - 1.6) * (z - 1.6));
  }
  hoodGeometry.computeVertexNormals();
  const hood = new THREE.Mesh(hoodGeometry, materials.paint);
  hood.position.set(0, -0.86, -2.3); // eye is 1.22 m up; hood crest sits ~0.5 m above the road
  group.add(hood);
  for (const side of [-1, 1]) {
    const pillar = new THREE.Mesh(new THREE.BoxGeometry(0.12, 1.5, 0.1), materials.trim);
    pillar.position.set(side * 0.98, 0.1, -0.78);
    pillar.rotation.set(-0.55, 0, side * -0.38);
    group.add(pillar);
  }
  const wheel = new THREE.Group();
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.19, 0.022, 12, 48), materials.trim);
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.065, 0.05, 20), materials.trim);
  hub.rotation.x = Math.PI / 2;
  const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.03, 0.02), materials.trim);
  wheel.add(rim, hub, spoke);
  wheel.position.set(0, -0.6, -0.48);
  wheel.rotation.x = -0.4;
  group.add(wheel);
  group.traverse((child) => { if (child.isMesh) child.renderOrder = 10; });
  return {group, wheel};
}

export class Vehicle {
  constructor(scene, camera, materials) {
    this.camera = camera;
    this.s = 0;
    this.speed = 0;
    this.accel = 0;
    this.pitch = 0;
    this.shake = 0;
    this.time = 0;
    this.rig = new THREE.Group();
    this.rig.add(camera);
    scene.add(this.rig);
    const {group, wheel} = cockpit(materials);
    this.wheel = wheel;
    camera.add(group);
    this.position = new THREE.Vector3();
  }

  reset(s = 0) {
    this.s = s;
    this.speed = 0;
    this.accel = 0;
    this.shake = 0;
  }

  get front() {
    return this.s + 2.3;
  }

  impulse(strength) {
    this.shake = Math.min(2.5, this.shake + strength);
  }

  // Hit something solid: stop dead just short of it.
  crash(blockS) {
    this.s = Math.min(this.s, blockS - 2.3);
    this.speed = 0;
    this.impulse(1.6);
  }

  update(dt, drive, limitS = Infinity) {
    let accel;
    if (drive) {
      const headroom = 1 - (this.speed / MAX_SPEED) ** 2;
      accel = ACCEL * Math.max(0, headroom) + (this.speed < 0.5 ? 0.6 : 0);
    } else {
      accel = this.speed > 0 ? -BRAKE : 0;
    }
    accel -= 0.0009 * this.speed * this.speed; // air drag
    this.speed = THREE.MathUtils.clamp(this.speed + accel * dt, 0, MAX_SPEED);
    if (!drive && this.speed < 0.15) this.speed = 0;
    this.accel += (accel - this.accel) * Math.min(1, dt * 4);
    const next = this.s + this.speed * dt;
    if (next >= limitS) {
      // Nose against an obstacle: the car cannot creep forward.
      this.s = Math.max(this.s, limitS);
      this.speed = 0;
    } else {
      this.s = next;
    }
    this.time += dt;
    this._pose(dt);
  }

  _pose(dt) {
    place(this.s, LANE, EYE_HEIGHT, this.position);
    this.rig.position.copy(this.position);
    const yaw = heading(this.s + 7);
    this.rig.rotation.set(0, yaw, 0);

    // Nose dives under braking, squats under throttle.
    const targetPitch = THREE.MathUtils.clamp(this.accel * 0.0045, -0.035, 0.02);
    this.pitch += (targetPitch - this.pitch) * Math.min(1, dt * 5);
    const t = this.time;
    const road = (this.speed / 21) * 0.0025;
    const vib = road * (Math.sin(t * 31) * 0.6 + Math.sin(t * 17.3) * 0.4);
    this.shake = Math.max(0, this.shake - dt * 1.8);
    const kick = this.shake * this.shake * 0.02;
    const jx = kick * (Math.sin(t * 47) + Math.sin(t * 29.7) * 0.5);
    const jy = kick * (Math.sin(t * 53.1) + Math.sin(t * 37.9) * 0.5);
    this.camera.position.set(jx * 2, vib * 6 + jy * 2, 0);
    this.camera.rotation.set(this.pitch + vib + jy, jx * 0.6, jx * 0.4);

    const turn = heading(this.s + 14) - heading(this.s);
    this.wheel.rotation.z = THREE.MathUtils.clamp(turn * 14, -1.2, 1.2);
  }
}
