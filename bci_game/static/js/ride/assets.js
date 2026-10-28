// Procedural textures and small models, built in code: no downloads.
import * as THREE from "../../vendor/three.module.min.js";

// Procedural textures painted on canvases — no image downloads, CSP-safe.

function canvas(width, height) {
  const element = document.createElement("canvas");
  element.width = width;
  element.height = height;
  return [element, element.getContext("2d")];
}

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function speckle(ctx, width, height, count, colors, sizeRange, random) {
  for (let i = 0; i < count; i += 1) {
    ctx.fillStyle = colors[Math.floor(random() * colors.length)];
    const size = sizeRange[0] + random() * (sizeRange[1] - sizeRange[0]);
    ctx.globalAlpha = 0.25 + random() * 0.6;
    ctx.fillRect(random() * width, random() * height, size, size);
  }
  ctx.globalAlpha = 1;
}

function finish(element, repeatU, repeatV, anisotropy) {
  const texture = new THREE.CanvasTexture(element);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeatU, repeatV);
  texture.anisotropy = anisotropy;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// Asphalt with worn edge lines and a dashed centre line. One tile = 12 m of road.
export function asphaltTexture(anisotropy) {
  const [element, ctx] = canvas(512, 1024);
  const random = rng(11);
  ctx.fillStyle = "#3b3c3e";
  ctx.fillRect(0, 0, 512, 1024);
  speckle(ctx, 512, 1024, 26000, ["#2c2d2f", "#4a4b4d", "#555658", "#323335"], [1, 3], random);
  // Tyre polish in each lane's wheel tracks.
  ctx.globalAlpha = 0.12;
  ctx.fillStyle = "#1f2022";
  for (const x of [62, 166, 310, 414]) ctx.fillRect(x, 0, 46, 1024);
  ctx.globalAlpha = 1;
  // Patch repairs.
  for (let i = 0; i < 5; i += 1) {
    ctx.fillStyle = random() > 0.5 ? "#303133" : "#454648";
    ctx.globalAlpha = 0.5;
    ctx.fillRect(random() * 420, random() * 960, 40 + random() * 80, 20 + random() * 70);
  }
  ctx.globalAlpha = 1;
  // Edge lines (solid white) and centre dashes (yellow, 3 m on 9 m off).
  const paint = (x, y, w, h, color) => {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
  };
  paint(10, 0, 10, 1024, "#d8d6cf");
  paint(492, 0, 10, 1024, "#d8d6cf");
  paint(247, 0, 8, 256, "#d2a63a");
  paint(257, 0, 8, 256, "#d2a63a");
  // Wear on the paint.
  speckle(ctx, 512, 1024, 4000, ["#3b3c3e"], [1, 2], random);
  return finish(element, 1, 1, anisotropy);
}

export function gravelTexture(anisotropy) {
  const [element, ctx] = canvas(256, 256);
  const random = rng(5);
  ctx.fillStyle = "#6b6252";
  ctx.fillRect(0, 0, 256, 256);
  speckle(ctx, 256, 256, 9000, ["#58503f", "#7e7462", "#8c8270", "#4a4335"], [1, 4], random);
  return finish(element, 1, 1, anisotropy);
}

export function grassTexture(anisotropy) {
  const [element, ctx] = canvas(512, 512);
  const random = rng(3);
  ctx.fillStyle = "#4d5a2c";
  ctx.fillRect(0, 0, 512, 512);
  // Soft macro patches so the tiling is less obvious.
  for (let i = 0; i < 40; i += 1) {
    const x = random() * 512;
    const y = random() * 512;
    const r = 30 + random() * 90;
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, r);
    const tone = ["rgba(98,104,52,0.5)", "rgba(60,72,34,0.5)", "rgba(112,98,60,0.35)"][i % 3];
    gradient.addColorStop(0, tone);
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }
  // Blades.
  for (let i = 0; i < 16000; i += 1) {
    const x = random() * 512;
    const y = random() * 512;
    ctx.strokeStyle = ["#3d4a20", "#5f6d33", "#6f7a3c", "#47552a", "#7c7444"][Math.floor(random() * 5)];
    ctx.globalAlpha = 0.4 + random() * 0.5;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + (random() - 0.5) * 3, y - 2 - random() * 5);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  return finish(element, 1, 1, anisotropy);
}

export function barkTexture() {
  const [element, ctx] = canvas(128, 256);
  const random = rng(9);
  ctx.fillStyle = "#4a3b2c";
  ctx.fillRect(0, 0, 128, 256);
  for (let i = 0; i < 140; i += 1) {
    ctx.strokeStyle = random() > 0.5 ? "#32271c" : "#5d4b38";
    ctx.lineWidth = 1 + random() * 3;
    const x = random() * 128;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.bezierCurveTo(x + 6, 80, x - 6, 170, x + 3, 256);
    ctx.stroke();
  }
  return finish(element, 1, 1, 1);
}

export function rockTexture() {
  const [element, ctx] = canvas(256, 256);
  const random = rng(21);
  ctx.fillStyle = "#77736b";
  ctx.fillRect(0, 0, 256, 256);
  speckle(ctx, 256, 256, 7000, ["#5f5b55", "#8a867e", "#6c6862", "#9a958c"], [1, 5], random);
  for (let i = 0; i < 30; i += 1) {
    ctx.strokeStyle = "rgba(40,38,35,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    let x = random() * 256;
    let y = random() * 256;
    ctx.moveTo(x, y);
    for (let k = 0; k < 5; k += 1) {
      x += (random() - 0.5) * 40;
      y += random() * 30;
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  return finish(element, 1, 1, 1);
}

// Soft radial glow used for lamps, brake lights, and sparks (additive sprites).
export function glowTexture() {
  const [element, ctx] = canvas(128, 128);
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.18, "rgba(255,255,255,0.75)");
  gradient.addColorStop(0.45, "rgba(255,255,255,0.18)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(element);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function dustTexture() {
  const [element, ctx] = canvas(64, 64);
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, "rgba(255,255,255,0.55)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(element);
}

// Shared geometry and small models (trees, deer, cars, traffic signal).

export function seeded(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return state / 4294967296;
  };
}

function jitter(geometry, amount, seed) {
  const random = seeded(seed);
  const position = geometry.attributes.position;
  const cache = new Map();
  for (let i = 0; i < position.count; i += 1) {
    const key = `${position.getX(i).toFixed(3)},${position.getY(i).toFixed(3)},${position.getZ(i).toFixed(3)}`;
    if (!cache.has(key)) cache.set(key, [(random() - 0.5) * amount, (random() - 0.5) * amount, (random() - 0.5) * amount]);
    const [dx, dy, dz] = cache.get(key);
    position.setXYZ(i, position.getX(i) + dx, position.getY(i) + dy, position.getZ(i) + dz);
  }
  geometry.computeVertexNormals();
  return geometry;
}

// A tiered spruce, 1 m tall at scale 1 (scale it to 8-22 m).
export function coniferGeometry() {
  const profile = [
    [0.0, 0.06], [0.42, 0.08], [0.25, 0.26], [0.36, 0.27], [0.19, 0.45], [0.29, 0.46],
    [0.13, 0.64], [0.21, 0.65], [0.07, 0.82], [0.12, 0.83], [0.0, 1.0],
  ].map(([r, y]) => new THREE.Vector2(r, y));
  return jitter(new THREE.LatheGeometry(profile, 10), 0.035, 3);
}

export function crownGeometry() {
  const geometry = new THREE.IcosahedronGeometry(1, 3);
  jitter(geometry, 0.22, 8);
  geometry.scale(1, 0.85, 1);
  return geometry;
}

export function trunkGeometry() {
  const geometry = new THREE.CylinderGeometry(0.6, 1, 1, 7, 1);
  geometry.translate(0, 0.5, 0);
  return geometry;
}

export function rockGeometry(seed, detail = 1) {
  return jitter(new THREE.DodecahedronGeometry(1, detail), 0.45, seed);
}

// One full tree as a group (used for the tree that falls).
export function makeTree(materials, height = 18, seed = 1) {
  const group = new THREE.Group();
  const trunk = new THREE.Mesh(trunkGeometry(), materials.bark);
  trunk.scale.set(0.38, height * 0.92, 0.38);
  const crown = new THREE.Mesh(coniferGeometry(), materials.heroConifer);
  crown.scale.set(height * 0.36, height, height * 0.36);
  crown.position.y = height * 0.05;
  crown.rotation.y = seeded(seed)() * Math.PI;
  for (const mesh of [trunk, crown]) { mesh.castShadow = true; mesh.receiveShadow = true; }
  group.add(trunk, crown);
  return group;
}

const glowCache = new Map();
export function glowSprite(texture, color, size) {
  const key = `${color}`;
  if (!glowCache.has(key)) {
    glowCache.set(key, new THREE.SpriteMaterial({
      map: texture, color, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false,
    }));
  }
  const sprite = new THREE.Sprite(glowCache.get(key).clone());
  sprite.scale.setScalar(size);
  return sprite;
}

// A mid-size hatchback. Faces -z (the direction of travel). Returns parts to animate.
export function makeCar(glow, paint = 0x8a1c1c) {
  const group = new THREE.Group();
  const body = new THREE.MeshPhysicalMaterial({color: paint, metalness: 0.6, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.12});
  const glass = new THREE.MeshPhysicalMaterial({color: 0x10151a, metalness: 0.2, roughness: 0.05, clearcoat: 1});
  const black = new THREE.MeshStandardMaterial({color: 0x111214, roughness: 0.7});
  const lower = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.62, 4.3), body);
  lower.position.y = 0.62;
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.58, 0.56, 2.3), glass);
  cabin.position.set(0, 1.2, 0.35);
  const roof = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.08, 2.0), body);
  roof.position.set(0, 1.5, 0.4);
  const bumper = new THREE.Mesh(new THREE.BoxGeometry(1.82, 0.22, 0.2), black);
  bumper.position.set(0, 0.42, 2.17);
  group.add(lower, cabin, roof, bumper);
  const wheelGeometry = new THREE.CylinderGeometry(0.34, 0.34, 0.24, 18);
  wheelGeometry.rotateZ(Math.PI / 2);
  for (const [x, z] of [[-0.86, -1.35], [0.86, -1.35], [-0.86, 1.35], [0.86, 1.35]]) {
    const wheel = new THREE.Mesh(wheelGeometry, black);
    wheel.position.set(x, 0.34, z);
    group.add(wheel);
  }
  const lampMaterial = new THREE.MeshStandardMaterial({color: 0x400808, emissive: 0xff1a0a, emissiveIntensity: 0.25});
  const lamps = [];
  const glows = [];
  for (const x of [-0.7, 0.7]) {
    const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.14, 0.04), lampMaterial);
    lamp.position.set(x, 0.82, 2.16);
    const halo = glowSprite(glow, 0xff2a14, 1.3);
    halo.position.set(x, 0.82, 2.3);
    halo.material.opacity = 0.0;
    lamps.push(lamp);
    glows.push(halo);
    group.add(lamp, halo);
  }
  const third = glowSprite(glow, 0xff2a14, 0.7);
  third.position.set(0, 1.5, 1.5);
  third.material.opacity = 0.0;
  glows.push(third);
  group.add(third);
  group.traverse((child) => { if (child.isMesh) { child.castShadow = true; child.receiveShadow = true; } });
  return {group, lampMaterial, glows};
}

// A white-tailed deer built from rounded primitives. Faces +x. Legs swing for a gallop.
export function makeDeer() {
  const group = new THREE.Group();
  const coat = new THREE.MeshStandardMaterial({color: 0x8a6440, roughness: 0.9});
  const belly = new THREE.MeshStandardMaterial({color: 0xcbb79a, roughness: 0.9});
  const dark = new THREE.MeshStandardMaterial({color: 0x2a1d14, roughness: 0.6});
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 12), coat);
  body.scale.set(1.45, 0.72, 0.62);
  body.position.y = 1.05;
  const chest = new THREE.Mesh(new THREE.SphereGeometry(0.3, 12, 10), belly);
  chest.position.set(0.52, 0.98, 0);
  chest.scale.set(1, 1.1, 0.95);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.2, 0.75, 10), coat);
  neck.position.set(0.72, 1.42, 0);
  neck.rotation.z = -0.55;
  const head = new THREE.Group();
  head.position.set(0.98, 1.78, 0);
  const skull = new THREE.Mesh(new THREE.SphereGeometry(0.15, 12, 10), coat);
  skull.scale.set(1.2, 0.95, 0.85);
  const snout = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.1, 0.3, 8), coat);
  snout.rotation.z = Math.PI / 2;
  snout.position.set(0.2, -0.05, 0);
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 6), dark);
  nose.position.set(0.35, -0.05, 0);
  head.add(skull, snout, nose);
  for (const z of [-0.09, 0.09]) {
    const ear = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.2, 6), coat);
    ear.position.set(-0.06, 0.15, z * 1.4);
    ear.rotation.x = z > 0 ? 0.5 : -0.5;
    head.add(ear);
  }
  const tail = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 6), belly);
  tail.position.set(-0.72, 1.18, 0);
  tail.scale.set(0.8, 1.5, 0.8);
  group.add(body, chest, neck, head, tail);
  const legs = [];
  for (const [x, z] of [[0.48, -0.16], [0.48, 0.16], [-0.5, -0.16], [-0.5, 0.16]]) {
    const pivot = new THREE.Group();
    pivot.position.set(x, 0.92, z);
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.06, 0.92, 6), coat);
    leg.position.y = -0.46;
    const hoof = new THREE.Mesh(new THREE.SphereGeometry(0.045, 6, 4), dark);
    hoof.position.y = -0.92;
    pivot.add(leg, hoof);
    legs.push(pivot);
    group.add(pivot);
  }
  group.traverse((child) => { if (child.isMesh) child.castShadow = true; });
  return {group, legs, head};
}

// Overhead mast-arm traffic signal. Origin at the pole base; the arm reaches toward -x.
export function makeSignal(glow) {
  const group = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({color: 0x5c6166, metalness: 0.7, roughness: 0.45});
  const housing = new THREE.MeshStandardMaterial({color: 0x22251f, roughness: 0.6});
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.17, 6.4, 12), metal);
  pole.position.y = 3.2;
  const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 5.2, 10), metal);
  arm.rotation.z = Math.PI / 2;
  arm.position.set(-2.5, 6.1, 0);
  group.add(pole, arm);
  const lamps = {};
  const heads = [];
  for (const offset of [-2.7, -0.6]) {
    const head = new THREE.Group();
    head.position.set(offset, 5.35, 0.12);
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.42, 1.18, 0.3), housing);
    const plate = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1.45, 0.03), housing);
    plate.position.z = -0.17;
    head.add(box, plate);
    [["red", 0.38, 0xff2b1c], ["amber", 0, 0xffa317], ["green", -0.38, 0x3dff8c]].forEach(([name, y, color]) => {
      const material = new THREE.MeshStandardMaterial({color: 0x111111, emissive: color, emissiveIntensity: 0.05});
      const lens = new THREE.Mesh(new THREE.CircleGeometry(0.13, 18), material);
      lens.position.set(0, y, 0.16);
      const visor = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.18, 14, 1, true, Math.PI * 0.5, Math.PI), housing);
      visor.rotation.x = Math.PI / 2;
      visor.position.set(0, y + 0.02, 0.24);
      const halo = glowSprite(glow, color, 1.6);
      halo.position.set(0, y, 0.3);
      halo.material.opacity = 0;
      head.add(lens, visor, halo);
      (lamps[name] ||= []).push({material, halo});
    });
    heads.push(head);
    group.add(head);
  }
  group.traverse((child) => { if (child.isMesh) child.castShadow = true; });
  return {group, lamps};
}
