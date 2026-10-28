// The world around the road: sky, ground, ridges, forest and roadside furniture.
import * as THREE from "../../vendor/three.module.min.js";
import {coniferGeometry, crownGeometry, rockGeometry, seeded, trunkGeometry} from "./assets.js";
import {heading, place, ROAD_HALF, SHOULDER} from "./road.js";

// Sky, sun, fog, ground, and distant ridges. Everything follows the car so the
// world feels endless while staying a fixed, small amount of geometry.

export const PRESETS = {
  golden: {
    zenith: 0x3d6fa8, horizon: 0xe9c79a, ground: 0x5a5a3a, fog: 0xcfb592, fogDensity: 0.0042,
    sun: 0xffd9a0, sunIntensity: 3.2, sunElevation: 9, sunAzimuth: -38, hemi: 1.1, clouds: 0.45,
    ridge: 0x7d8aa3,
  },
  overcast: {
    zenith: 0x6d7884, horizon: 0xaab1b6, ground: 0x3f4434, fog: 0xa4abaf, fogDensity: 0.0062,
    sun: 0xf2f1ec, sunIntensity: 1.8, sunElevation: 38, sunAzimuth: 20, hemi: 1.15, clouds: 0.85,
    ridge: 0x8f979f,
  },
};

const SKY_VERTEX = /* glsl */ `
  varying vec3 vDir;
  void main() {
    vDir = normalize(position);
    vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_Position = p.xyww;
  }
`;

const SKY_FRAGMENT = /* glsl */ `
  uniform vec3 zenith; uniform vec3 horizon; uniform vec3 sunColor; uniform vec3 sunDir;
  uniform float cloudAmount; uniform float time;
  varying vec3 vDir;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
  }
  float fbm(vec2 p) { float v = 0.0; float a = 0.5; for (int i = 0; i < 5; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; } return v; }
  void main() {
    vec3 dir = normalize(vDir);
    float h = max(dir.y, 0.0);
    vec3 color = mix(horizon, zenith, pow(h, 0.55));
    float sun = max(dot(dir, sunDir), 0.0);
    color += sunColor * (pow(sun, 900.0) * 6.0 + pow(sun, 18.0) * 0.35 + pow(sun, 3.0) * 0.08);
    if (dir.y > 0.0) {
      vec2 uv = dir.xz / (dir.y + 0.12) * 1.4 + vec2(time * 0.004, 0.0);
      float c = smoothstep(1.0 - cloudAmount, 1.25 - cloudAmount * 0.5, fbm(uv));
      vec3 cloud = mix(horizon * 1.05, vec3(1.0), 0.55) + sunColor * pow(sun, 6.0) * 0.4;
      color = mix(color, cloud, c * smoothstep(0.0, 0.18, dir.y) * 0.85);
    }
    color = mix(color, horizon, smoothstep(0.08, -0.02, dir.y));
    gl_FragColor = vec4(color, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

function ridgeGeometry(radius, seed) {
  const segments = 360;
  const positions = [];
  const index = [];
  let state = seed;
  const random = () => { state = (state * 16807) % 2147483647; return state / 2147483647; };
  const peaks = Array.from({length: 18}, () => [random() * Math.PI * 2, 40 + random() * 160, 0.05 + random() * 0.18]);
  for (let i = 0; i <= segments; i += 1) {
    const angle = (i / segments) * Math.PI * 2;
    let height = 30;
    for (const [center, size, width] of peaks) {
      const distance = Math.atan2(Math.sin(angle - center), Math.cos(angle - center));
      height += size * Math.exp(-(distance * distance) / (width * width));
    }
    height += 12 * Math.sin(angle * 23) + 6 * Math.sin(angle * 61);
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    positions.push(x, -20, z, x, height, z);
    if (i < segments) {
      const a = i * 2;
      index.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(index);
  return geometry;
}

export class Environment {
  constructor(scene, renderer, textures, presetName = "golden") {
    this.name = PRESETS[presetName] ? presetName : "golden";
    const preset = PRESETS[this.name];
    this.preset = preset;
    scene.fog = new THREE.FogExp2(preset.fog, preset.fogDensity);
    scene.background = new THREE.Color(preset.horizon);

    const elevation = THREE.MathUtils.degToRad(preset.sunElevation);
    const azimuth = THREE.MathUtils.degToRad(preset.sunAzimuth);
    this.sunDirection = new THREE.Vector3(
      Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation), -Math.cos(elevation) * Math.cos(azimuth),
    ).normalize();

    this.skyUniforms = {
      zenith: {value: new THREE.Color(preset.zenith)},
      horizon: {value: new THREE.Color(preset.horizon)},
      sunColor: {value: new THREE.Color(preset.sun)},
      sunDir: {value: this.sunDirection},
      cloudAmount: {value: preset.clouds},
      time: {value: 0},
    };
    this.sky = new THREE.Mesh(
      new THREE.SphereGeometry(3000, 32, 16),
      new THREE.ShaderMaterial({
        uniforms: this.skyUniforms, vertexShader: SKY_VERTEX, fragmentShader: SKY_FRAGMENT,
        side: THREE.BackSide, depthWrite: false, fog: false,
      }),
    );
    this.sky.renderOrder = -1;
    scene.add(this.sky);

    this.hemi = new THREE.HemisphereLight(preset.horizon, preset.ground, preset.hemi);
    scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(preset.sun, preset.sunIntensity);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const shadow = this.sun.shadow.camera;
    shadow.left = -70; shadow.right = 70; shadow.top = 70; shadow.bottom = -70;
    shadow.near = 1; shadow.far = 400;
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.6;
    scene.add(this.sun, this.sun.target);

    textures.grass.repeat.set(3000 / 9, 3000 / 9);
    this.ground = new THREE.Mesh(
      new THREE.PlaneGeometry(3000, 3000),
      new THREE.MeshStandardMaterial({map: textures.grass, roughness: 1, color: 0xd6dcc4}),
    );
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    scene.add(this.ground);

    const ridgeMaterial = (color, fogMix) => new THREE.MeshBasicMaterial({
      color: new THREE.Color(color).lerp(new THREE.Color(preset.fog), fogMix), fog: false, side: THREE.DoubleSide,
    });
    this.ridges = new THREE.Group();
    this.ridges.add(new THREE.Mesh(ridgeGeometry(2400, 7), ridgeMaterial(preset.ridge, 0.45)));
    const near = new THREE.Mesh(ridgeGeometry(1700, 19), ridgeMaterial(0x4f5d4a, 0.55));
    near.scale.y = 0.55;
    this.ridges.add(near);
    scene.add(this.ridges);

    // Image-based lighting from the sky so paint and glass pick up reflections.
    const pmrem = new THREE.PMREMGenerator(renderer);
    const skyScene = new THREE.Scene();
    skyScene.add(this.sky.clone());
    this.envMap = pmrem.fromScene(skyScene, 0.02).texture;
    scene.environment = this.envMap;
    pmrem.dispose();
  }

  update(cameraPosition, elapsed) {
    this.sky.position.copy(cameraPosition);
    this.ridges.position.set(cameraPosition.x, 0, cameraPosition.z);
    const tile = 9;
    this.ground.position.set(cameraPosition.x, 0, cameraPosition.z);
    const map = this.ground.material.map;
    map.offset.set(cameraPosition.x / tile, -cameraPosition.z / tile);
    this.sun.position.copy(cameraPosition).addScaledVector(this.sunDirection, 200);
    this.sun.target.position.copy(cameraPosition);
    this.skyUniforms.time.value = elapsed;
  }
}

// Roadside forest, delineator posts, and power lines, recycled ahead of the car.

const BEHIND = 30;
const SPAN = 380;
const POLE_GAP = 48;
const POST_GAP = 25;

const dummy = new THREE.Object3D();
const color = new THREE.Color();

export class Scenery {
  constructor(scene, materials) {
    this.random = seeded(4242);
    this.clearings = [];
    this.trees = this._forest(scene, materials, 1500, 380);
    this.bushes = this._bushes(scene, materials, 420);
    this.posts = this._posts(scene);
    this._poles(scene, materials);
  }

  // -- forest ------------------------------------------------------------
  _forest(scene, materials, conifers, broadleaf) {
    const total = conifers + broadleaf;
    const trunks = new THREE.InstancedMesh(trunkGeometry(), materials.bark, total);
    const spruce = new THREE.InstancedMesh(coniferGeometry(), materials.conifer, conifers);
    const crowns = new THREE.InstancedMesh(crownGeometry(), materials.broadleaf, broadleaf);
    for (const mesh of [trunks, spruce, crowns]) {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = false;
      scene.add(mesh);
    }
    const items = [];
    for (let i = 0; i < total; i += 1) {
      items.push({index: i, conifer: i < conifers, s: -BEHIND + this.random() * SPAN, l: 0, h: 0, yaw: 0});
    }
    const forest = {trunks, spruce, crowns, items, conifers};
    for (const item of items) this._placeTree(forest, item, item.s);
    for (const mesh of [trunks, spruce, crowns]) mesh.instanceMatrix.needsUpdate = true;
    for (const mesh of [spruce, crowns]) if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    return forest;
  }

  _blocked(s) {
    return this.clearings.some(([from, to]) => s >= from && s <= to);
  }

  _placeTree(forest, item, s) {
    let target = s;
    while (this._blocked(target)) target += 6 + this.random() * 6;
    item.s = target;
    const side = this.random() < 0.5 ? -1 : 1;
    const nearEdge = ROAD_HALF + SHOULDER + (item.conifer ? 4.5 : 3.5);
    item.l = side * (nearEdge + Math.pow(this.random(), 1.5) * 90);
    const near = Math.min(1, (Math.abs(item.l) - nearEdge) / 12);
    item.h = (item.conifer ? 9 + this.random() * 14 : 7 + this.random() * 8) * (0.65 + 0.35 * near);
    item.yaw = this.random() * Math.PI * 2;
    const base = place(item.s, item.l, 0);
    const lean = (this.random() - 0.5) * 0.06;
    dummy.position.copy(base);
    dummy.rotation.set(lean, item.yaw, lean);
    if (item.conifer) {
      dummy.scale.set(0.3, item.h * 0.95, 0.3);
      dummy.updateMatrix();
      forest.trunks.setMatrixAt(item.index, dummy.matrix);
      dummy.scale.set(item.h * (0.3 + this.random() * 0.1), item.h, item.h * 0.34);
      dummy.position.y = item.h * 0.05;
      dummy.updateMatrix();
      forest.spruce.setMatrixAt(item.index, dummy.matrix);
      color.setHSL(0.29 + this.random() * 0.07, 0.30 + this.random() * 0.14, 0.035 + this.random() * 0.035);
      forest.spruce.setColorAt(item.index, color);
    } else {
      dummy.scale.set(0.28, item.h * 0.55, 0.28);
      dummy.updateMatrix();
      forest.trunks.setMatrixAt(item.index, dummy.matrix);
      const crown = item.h * (0.32 + this.random() * 0.1);
      dummy.position.y = item.h * 0.62;
      dummy.scale.set(crown, crown * (0.9 + this.random() * 0.3), crown);
      dummy.updateMatrix();
      forest.crowns.setMatrixAt(item.index - forest.conifers, dummy.matrix);
      color.setHSL(0.20 + this.random() * 0.08, 0.38 + this.random() * 0.18, 0.045 + this.random() * 0.04);
      forest.crowns.setColorAt(item.index - forest.conifers, color);
    }
  }

  // -- low bushes and rocks along the verge -------------------------------
  _bushes(scene, materials, count) {
    const mesh = new THREE.InstancedMesh(rockGeometry(31, 1), materials.bush, count);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    mesh.frustumCulled = false;
    scene.add(mesh);
    const items = Array.from({length: count}, (_, index) => ({index, s: -BEHIND + this.random() * SPAN}));
    const bushes = {mesh, items};
    for (const item of items) this._placeBush(bushes, item, item.s);
    mesh.instanceMatrix.needsUpdate = true;
    return bushes;
  }

  _placeBush(bushes, item, s) {
    item.s = s;
    const side = this.random() < 0.5 ? -1 : 1;
    const l = side * (ROAD_HALF + SHOULDER + 0.6 + this.random() * 9);
    const size = 0.3 + this.random() * 0.8;
    dummy.position.copy(place(s, l, size * 0.25));
    dummy.rotation.set(0, this.random() * 6, 0);
    dummy.scale.set(size * 1.3, size * 0.7, size);
    dummy.updateMatrix();
    bushes.mesh.setMatrixAt(item.index, dummy.matrix);
    color.setHSL(0.24 + this.random() * 0.06, 0.35, 0.035 + this.random() * 0.04);
    bushes.mesh.setColorAt(item.index, color);
  }

  // -- white delineator posts --------------------------------------------
  _posts(scene) {
    const count = Math.ceil(SPAN / POST_GAP) * 2;
    const geometry = new THREE.BoxGeometry(0.11, 1.05, 0.11);
    geometry.translate(0, 0.52, 0);
    const mesh = new THREE.InstancedMesh(geometry, new THREE.MeshStandardMaterial({color: 0xe8e6df, roughness: 0.5}), count);
    const reflector = new THREE.BoxGeometry(0.08, 0.16, 0.02);
    reflector.translate(0, 0.86, 0.06);
    const glints = new THREE.InstancedMesh(reflector, new THREE.MeshStandardMaterial({color: 0x331100, emissive: 0xff8a1a, emissiveIntensity: 0.9}), count);
    for (const m of [mesh, glints]) { m.frustumCulled = false; m.castShadow = true; scene.add(m); }
    return {mesh, glints, count, first: null};
  }

  _updatePosts(carS) {
    const first = Math.floor((carS - BEHIND) / POST_GAP);
    if (first === this.posts.first) return;
    this.posts.first = first;
    for (let i = 0; i < this.posts.count / 2; i += 1) {
      const s = (first + i) * POST_GAP;
      [-1, 1].forEach((side, k) => {
        dummy.position.copy(place(s, side * (ROAD_HALF + SHOULDER + 0.35), 0));
        dummy.rotation.set(0, heading(s), 0);
        dummy.scale.setScalar(this._blocked(s) ? 0 : 1);
        dummy.updateMatrix();
        this.posts.mesh.setMatrixAt(i * 2 + k, dummy.matrix);
        this.posts.glints.setMatrixAt(i * 2 + k, dummy.matrix);
      });
    }
    this.posts.mesh.instanceMatrix.needsUpdate = true;
    this.posts.glints.instanceMatrix.needsUpdate = true;
  }

  // -- wooden power poles with sagging lines ------------------------------
  _poles(scene, materials) {
    const count = Math.ceil(SPAN / POLE_GAP) + 1;
    const pole = new THREE.CylinderGeometry(0.13, 0.17, 10, 8);
    pole.translate(0, 5, 0);
    const arm = new THREE.BoxGeometry(2.2, 0.12, 0.12);
    arm.translate(0, 9.3, 0);
    this.poleMesh = new THREE.InstancedMesh(pole, materials.bark, count);
    this.armMesh = new THREE.InstancedMesh(arm, materials.bark, count);
    for (const m of [this.poleMesh, this.armMesh]) { m.castShadow = true; m.frustumCulled = false; scene.add(m); }
    this.wireGeometry = new THREE.BufferGeometry();
    this.wireGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array((count - 1) * 3 * 12 * 2 * 3), 3));
    this.wires = new THREE.LineSegments(this.wireGeometry, new THREE.LineBasicMaterial({color: 0x1d1f20, transparent: true, opacity: 0.8}));
    this.wires.frustumCulled = false;
    scene.add(this.wires);
    this.poleCount = count;
    this.firstPole = null;
  }

  _updatePoles(carS) {
    const first = Math.floor((carS - BEHIND) / POLE_GAP);
    if (first === this.firstPole) return;
    this.firstPole = first;
    const tops = [];
    for (let i = 0; i < this.poleCount; i += 1) {
      const s = (first + i) * POLE_GAP;
      const base = place(s, -(ROAD_HALF + SHOULDER + 3.2), 0);
      const yaw = heading(s);
      dummy.position.copy(base);
      dummy.rotation.set(0, yaw, 0);
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      this.poleMesh.setMatrixAt(i, dummy.matrix);
      this.armMesh.setMatrixAt(i, dummy.matrix);
      tops.push({base, yaw});
    }
    const positions = this.wireGeometry.attributes.position.array;
    let k = 0;
    const offset = new THREE.Vector3();
    for (let i = 0; i < tops.length - 1; i += 1) {
      for (const lateral of [-0.95, 0, 0.95]) {
        for (let seg = 0; seg < 12; seg += 1) {
          for (const t of [seg / 12, (seg + 1) / 12]) {
            const a = tops[i];
            const b = tops[i + 1];
            offset.set(lateral, 0, 0).applyAxisAngle(THREE.Object3D.DEFAULT_UP, a.yaw);
            const x = a.base.x + (b.base.x - a.base.x) * t + offset.x;
            const z = a.base.z + (b.base.z - a.base.z) * t + offset.z;
            const y = 9.4 - 4 * 0.9 * t * (1 - t);
            positions[k++] = x; positions[k++] = y; positions[k++] = z;
          }
        }
      }
    }
    this.wireGeometry.attributes.position.needsUpdate = true;
    this.poleMesh.instanceMatrix.needsUpdate = true;
    this.armMesh.instanceMatrix.needsUpdate = true;
  }

  // Keep a stretch of road free of trees (intersections, cliffs).
  reserve(from, to) {
    this.clearings.push([from, to]);
    let dirty = false;
    for (const item of this.trees.items) {
      if (item.s >= from && item.s <= to && Math.abs(item.l) < 160) {
        this._placeTree(this.trees, item, to + this.random() * 8);
        dirty = true;
      }
    }
    if (dirty) this._flagTrees();
    this.posts.first = null;
  }

  release(from, to) {
    this.clearings = this.clearings.filter(([a, b]) => a !== from || b !== to);
  }

  _flagTrees() {
    const {trunks, spruce, crowns} = this.trees;
    for (const mesh of [trunks, spruce, crowns]) mesh.instanceMatrix.needsUpdate = true;
    for (const mesh of [spruce, crowns]) if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  update(carS) {
    let dirty = false;
    for (const item of this.trees.items) {
      if (item.s < carS - BEHIND) {
        this._placeTree(this.trees, item, item.s + SPAN);
        dirty = true;
      }
    }
    if (dirty) this._flagTrees();
    let bushDirty = false;
    for (const item of this.bushes.items) {
      if (item.s < carS - BEHIND) {
        this._placeBush(this.bushes, item, item.s + SPAN);
        bushDirty = true;
      }
    }
    if (bushDirty) {
      this.bushes.mesh.instanceMatrix.needsUpdate = true;
      if (this.bushes.mesh.instanceColor) this.bushes.mesh.instanceColor.needsUpdate = true;
    }
    this._updatePosts(carS);
    this._updatePoles(carS);
  }
}
