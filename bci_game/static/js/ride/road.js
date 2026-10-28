// The road: an endless gently curving two-lane country road.
// Distance along the road is `s` (metres); `l` is the lateral offset (+ = right).
import * as THREE from "../../vendor/three.module.min.js";

export const ROAD_HALF = 3.7;      // asphalt half width
export const SHOULDER = 1.6;       // gravel shoulder width
export const LANE = 1.75;          // centre of our (right-hand) lane
const ROW = 2;                     // metres between ribbon rows
const BEHIND = 24;
const AHEAD = 420;
const TILE = 12;                   // metres of road per asphalt texture tile

export function roadX(s) {
  return 22 * Math.sin(s / 310) + 10 * Math.sin(s / 131 + 1.1) + 3.5 * Math.sin(s / 57 + 0.4);
}

export function roadSlope(s) {
  return (22 / 310) * Math.cos(s / 310) + (10 / 131) * Math.cos(s / 131 + 1.1) + (3.5 / 57) * Math.cos(s / 57 + 0.4);
}

// World position of a point at distance s and lateral offset l.
export function place(s, l, y = 0, target = new THREE.Vector3()) {
  const slope = roadSlope(s);
  const norm = Math.hypot(1, slope);
  return target.set(roadX(s) + l / norm, y, -s + (l * slope) / norm);
}

// Yaw (radians about +y) of the road's forward direction at s.
export function heading(s) {
  return Math.atan2(-roadSlope(s), 1);
}

function ribbon(rows, material) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(rows * 2 * 3), 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(new Float32Array(rows * 2 * 2), 2));
  geometry.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(rows * 2 * 3), 3));
  const index = [];
  for (let r = 0; r < rows - 1; r += 1) {
    const a = r * 2;
    index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); // counter-clockwise from above
  }
  geometry.setIndex(index);
  const normals = geometry.attributes.normal.array;
  for (let i = 1; i < normals.length; i += 3) normals[i] = 1;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.receiveShadow = true;
  return mesh;
}

export class Road {
  constructor(textures) {
    this.rows = Math.ceil((BEHIND + AHEAD) / ROW) + 1;
    this.group = new THREE.Group();
    const asphalt = new THREE.MeshStandardMaterial({
      map: textures.asphalt, roughness: 0.92, metalness: 0.0,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    });
    const gravel = new THREE.MeshStandardMaterial({
      map: textures.gravel, roughness: 1.0,
      polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
    });
    this.asphalt = ribbon(this.rows, asphalt);
    this.left = ribbon(this.rows, gravel);
    this.right = ribbon(this.rows, gravel);
    this.group.add(this.asphalt, this.left, this.right);
    this.firstRow = null;
    this._point = new THREE.Vector3();
  }

  _fill(mesh, firstS, inner, outer, y, uScale, vScale) {
    const position = mesh.geometry.attributes.position.array;
    const uv = mesh.geometry.attributes.uv.array;
    for (let r = 0; r < this.rows; r += 1) {
      const s = firstS + r * ROW;
      place(s, inner, y, this._point);
      position.set([this._point.x, this._point.y, this._point.z], r * 6);
      place(s, outer, y, this._point);
      position.set([this._point.x, this._point.y, this._point.z], r * 6 + 3);
      uv.set([0, s / vScale, uScale, s / vScale], r * 4);
    }
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.attributes.uv.needsUpdate = true;
    mesh.geometry.computeBoundingSphere();
  }

  update(carS) {
    const firstRow = Math.floor((carS - BEHIND) / ROW);
    if (firstRow === this.firstRow) return;
    this.firstRow = firstRow;
    const firstS = firstRow * ROW;
    this._fill(this.asphalt, firstS, -ROAD_HALF, ROAD_HALF, 0.03, 1, TILE);
    this._fill(this.left, firstS, -ROAD_HALF - SHOULDER, -ROAD_HALF, 0.015, 1, 4);
    this._fill(this.right, firstS, ROAD_HALF, ROAD_HALF + SHOULDER, 0.015, 1, 4);
  }
}
