// Krea2 Regional Multi-LoRA (By Fedor) - dynamic region rows.
//
// Adds an "+ Add Region" button and, per region, a LoRA combo + strength +
// enable toggle + remove button. All rows are serialized into the node's
// `regions_json` widget (the Python side's source of truth), so the graph
// still round-trips through save/load and the API.
//
// Auto-sync: when a bounding-box builder is wired into the bboxes input, the
// region row count automatically follows the number of boxes drawn there.
// Existing LoRA assignments are preserved; only the count changes.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Krea2RegionalMultiLoRA";
const JSON_WIDGET = "regions_json";

let LORA_LIST = ["None"];

async function ensureLoraList() {
  if (LORA_LIST.length > 1) return LORA_LIST;
  try {
    const resp = await api.fetchApi("/object_info/LoraLoader");
    const info = await resp.json();
    const names = info?.LoraLoader?.input?.required?.lora_name?.[0];
    if (Array.isArray(names) && names.length) {
      LORA_LIST = ["None", ...names.filter((n) => n !== "None")];
    }
  } catch (e) {
    console.warn("[Krea2RegionalMultiLoRA] could not fetch lora list:", e);
  }
  return LORA_LIST;
}

function readRegions(node) {
  const w = node.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return [];
  try {
    const parsed = JSON.parse(w.value || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function writeRegions(node, regions) {
  const w = node.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return;
  w.value = JSON.stringify(regions, null, 2);
  if (w.inputEl) w.inputEl.value = w.value;
}

function markTransient(w) {
  w.__k2region = true;
  w.serialize = false;
  if (!w.options) w.options = {};
  w.options.serialize = false;
  return w;
}

// ---------------------------------------------------------------------------
// Bbox auto-sync
// ---------------------------------------------------------------------------

// Return the number of boxes in the node wired to our bboxes input, or null
// if nothing is wired or the source can't be parsed.
function getBboxCount(node) {
  const bboxInput = node.inputs?.find((i) => i.name === "bboxes");
  if (!bboxInput?.link) return null;

  const linkInfo = node.graph?.links?.[bboxInput.link];
  if (!linkInfo) return null;

  const srcNode = node.graph?.getNodeById(linkInfo.origin_id);
  if (!srcNode) return null;

  // Find the source widget whose value parses to an array of box objects
  // (objects with spatial keys x / x0 / w / width).
  for (const w of srcNode.widgets || []) {
    if (typeof w.value !== "string") continue;
    try {
      const parsed = JSON.parse(w.value);
      if (
        Array.isArray(parsed) &&
        parsed.length > 0 &&
        typeof parsed[0] === "object" &&
        ("x" in parsed[0] || "x0" in parsed[0] || "width" in parsed[0] || "w" in parsed[0])
      ) {
        return parsed.length;
      }
    } catch (_) {}
  }
  return null;
}

// Adjust the regions array so its length matches targetCount, preserving all
// existing LoRA assignments. New rows get sensible defaults.
function syncRegionCount(node, targetCount) {
  const regions = readRegions(node);
  if (regions.length === targetCount) return;

  if (regions.length < targetCount) {
    while (regions.length < targetCount) {
      regions.push({ lora: "None", strength: 1.1, enable: true });
    }
  } else {
    regions.splice(targetCount);
  }

  writeRegions(node, regions);
  rebuildRows(node);
}

// ---------------------------------------------------------------------------
// Row rendering
// ---------------------------------------------------------------------------

function rebuildRows(node) {
  if (node.widgets) {
    node.widgets = node.widgets.filter((w) => !w.__k2region);
  }

  const regions = readRegions(node);

  regions.forEach((region, idx) => {
    const enableW = node.addWidget(
      "toggle",
      `region ${idx + 1} enabled`,
      region.enable !== false,
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].enable = v; writeRegions(node, r); }
      },
      { on: "on", off: "off" }
    );
    markTransient(enableW);

    const loraW = node.addWidget(
      "combo",
      `region ${idx + 1} lora`,
      region.lora || "None",
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].lora = v; writeRegions(node, r); }
      },
      { values: LORA_LIST }
    );
    markTransient(loraW);

    const strW = node.addWidget(
      "number",
      `region ${idx + 1} strength`,
      typeof region.strength === "number" ? region.strength : 1.1,
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].strength = v; writeRegions(node, r); }
      },
      { min: -10.0, max: 10.0, step: 0.1, precision: 2 }
    );
    markTransient(strW);

    const rmW = node.addWidget("button", `  remove region ${idx + 1}`, null, () => {
      const r = readRegions(node);
      r.splice(idx, 1);
      writeRegions(node, r);
      rebuildRows(node);
      node.setDirtyCanvas(true, true);
    });
    markTransient(rmW);
  });

  node.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "krea2.RegionalMultiLoRA",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    await ensureLoraList();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      this.__k2lastBboxCount = null;

      const addBtn = this.addWidget("button", "+ Add Region", null, () => {
        const regions = readRegions(this);
        regions.push({ lora: "None", strength: 1.1, enable: true });
        writeRegions(this, regions);
        rebuildRows(this);
      });
      addBtn.__k2add = true;

      rebuildRows(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (o) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      this.__k2lastBboxCount = null;
      setTimeout(() => rebuildRows(this), 0);
      return r;
    };

    // Immediate sync when the bboxes wire is connected or removed.
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
      const r = onConnectionsChange
        ? onConnectionsChange.apply(this, arguments)
        : undefined;

      const bboxIdx = this.inputs?.findIndex((i) => i.name === "bboxes");
      if (index === bboxIdx) {
        this.__k2lastBboxCount = null;
        setTimeout(() => {
          const count = getBboxCount(this);
          if (count !== null) {
            this.__k2lastBboxCount = count;
            syncRegionCount(this, count);
          }
        }, 50);
      }
      return r;
    };

    // Detect box add/remove in the connected builder. The __k2lastBboxCount
    // cache means we only act when the count actually changes, so the
    // per-frame cost is just two property lookups.
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (onDrawForeground) onDrawForeground.apply(this, arguments);

      const count = getBboxCount(this);
      if (count !== null && count !== this.__k2lastBboxCount) {
        this.__k2lastBboxCount = count;
        syncRegionCount(this, count);
      }
    };
  },
});
