// node --test tests/road.test.mjs
import assert from "node:assert/strict";
import test from "node:test";
import {heading, place, roadSlope, roadX} from "../bci_game/static/js/ride/road.js";

test("slope is the derivative of the centre line", () => {
  for (const s of [0, 137, 912, 4800]) {
    const numeric = (roadX(s + 0.01) - roadX(s - 0.01)) / 0.02;
    assert.ok(Math.abs(numeric - roadSlope(s)) < 1e-4);
  }
});

test("lateral offsets are perpendicular to the road and the right distance", () => {
  for (const s of [10, 333, 2500]) {
    const centre = place(s, 0);
    const right = place(s, 3.5);
    const ahead = place(s + 0.5, 0);
    const lateral = right.clone().sub(centre);
    const forward = ahead.clone().sub(centre).normalize();
    assert.ok(Math.abs(lateral.length() - 3.5) < 1e-6);
    assert.ok(Math.abs(lateral.dot(forward)) < 1e-2);
  }
});

test("heading points the camera down the road", () => {
  for (const s of [0, 640, 1700]) {
    const yaw = heading(s);
    const look = {x: -Math.sin(yaw), z: -Math.cos(yaw)};
    const ahead = place(s + 1, 0).sub(place(s, 0)).normalize();
    assert.ok(look.x * ahead.x + look.z * ahead.z > 0.9999);
  }
});
