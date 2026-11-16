import assert from "node:assert/strict";

import { normalizeChordEvents } from "../../static/js/chordTrack.js";

assert.deepEqual(
  normalizeChordEvents([
    { start: 2, end: 4, symbol: "G7", confidence: 0.8 },
    { start: 0, end: 3, symbol: "Cmaj7", confidence: 1.2 },
    { start: 4, end: 8, symbol: "N.C.", confidence: 0.4 },
    { start: 1, end: 2, symbol: "bad" },
    { start: 1, end: 1, symbol: "Dm7" },
  ], 6),
  [
    { start: 0, end: 3, symbol: "Cmaj7", confidence: 1 },
    { start: 3, end: 4, symbol: "G7", confidence: 0.8 },
    { start: 4, end: 6, symbol: "N.C.", confidence: 0.4 },
  ],
);

assert.deepEqual(normalizeChordEvents(null, 60), []);
assert.deepEqual(normalizeChordEvents([{ start: -1, end: 1, symbol: "C" }], 60), []);

console.log("Chord track normalization tests passed");
