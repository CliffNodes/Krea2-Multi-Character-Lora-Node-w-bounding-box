// Krea2 Reference Lock — Multi (By Fedor)
// Dynamic IMAGE inputs: reference_0, reference_1, ... one per bounding box.
// reference_i maps to box i (same order as the MultiLoRA regions), so you can
// see at a glance which image each box/LoRA references. Slot count follows the
// wired bbox source when present, otherwise grows as you connect references.

import { app } from "../../scripts/app.js";

const NODE = "Krea2ReferenceLockMulti";
const PREFIX = "reference_";
const TYPE = "IMAGE";

function idxOf(inp) {
  if (!inp || !inp.name || !inp.name.startsWith(PREFIX)) return null;
  const n = parseInt(inp.name.slice(PREFIX.length), 10);
  return isNaN(n) ? null : n;
}

// Count boxes from a connected BOUNDING_BOX source (same heuristic as the
// MultiLoRA node): find the source node's JSON-string widget of box objects.
function getBboxCount(node) {
  const bboxInput = (node.inputs || []).find((i) => i.name === "bboxes");
  if (!bboxInput || bboxInput.link == null) return null;
  const link = node.graph && node.graph.links ? node.graph.links[bboxInput.link] : null;
  if (!link) return null;
  const src = node.graph.getNodeById(link.origin_id);
  if (!src) return null;
  for (const w of src.widgets || []) {
    if (typeof w.value !== "string") continue;
    try {
      const parsed = JSON.parse(w.value);
      if (
        Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === "object" &&
        ("x" in parsed[0] || "x0" in parsed[0] || "width" in parsed[0] || "w" in parsed[0])
      ) {
        return parsed.length;
      }
    } catch (_) {}
  }
  return null;
}

function highestConnectedRef(node) {
  let hi = -1;
  for (const inp of node.inputs || []) {
    const n = idxOf(inp);
    if (n !== null && inp.link != null) hi = Math.max(hi, n);
  }
  return hi;
}

function desiredCount(node) {
  const bb = getBboxCount(node);
  const connectedTop = highestConnectedRef(node) + 1; // keep all connected slots
  if (bb !== null) return Math.max(bb, connectedTop);
  return Math.max(1, connectedTop + 1); // no bbox info: one trailing empty
}

function ensureRefInputs(node, count) {
  count = Math.max(1, count);
  const have = {};
  for (const inp of node.inputs || []) {
    const n = idxOf(inp);
    if (n !== null) have[n] = inp;
  }
  for (let i = 0; i < count; i++) {
    if (!have[i]) node.addInput(PREFIX + i, TYPE);
  }
  for (let i = node.inputs.length - 1; i >= 0; i--) {
    const inp = node.inputs[i];
    const n = idxOf(inp);
    if (n !== null && n >= count && inp.link == null) node.removeInput(i);
  }
  for (const inp of node.inputs || []) {
    const n = idxOf(inp);
    if (n !== null) inp.label = `${PREFIX}${n} (box ${n})`;
  }
}

app.registerExtension({
  name: "Krea2.ReferenceLockMulti",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      ensureRefInputs(this, desiredCount(this));
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
      const self = this;
      setTimeout(() => {
        ensureRefInputs(self, desiredCount(self));
        self.setDirtyCanvas(true, true);
      }, 0);
      return r;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
      if (onDrawForeground) onDrawForeground.apply(this, arguments);
      const want = desiredCount(this);
      if (want !== this.__k2refLast) {
        this.__k2refLast = want;
        ensureRefInputs(this, want);
      }
    };
  },
});
