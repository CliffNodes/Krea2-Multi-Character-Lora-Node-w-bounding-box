// Krea2 Character (By Fedor) - chain -> regions_json live sync.
//
// The server treats a wired chain as authoritative at render time, but that is
// invisible until you hit Queue: the regional node still shows whatever rows
// were typed before, so there is no way to tell a working chain from a dead
// one. Here the chain is walked in the browser and written straight into
// regions_json, so the region rows ARE the chain - LoRA names, strengths and
// prompts all render in the existing row UI the moment you wire or edit a
// character.
//
// Reference images stay on their sockets: those are live IMAGE tensors with no
// filename, so only the server can resolve them. The rows leave ref_image empty
// and the badge reports how many characters carry one.

import { app } from "../../scripts/app.js";

const CHARACTER_NODE = "Krea2Character";
const REGIONAL_NODES = [
  "Krea2RegionalMultiLoRAV3",
  "Krea2RegionalMultiLoRAV9",
  "Krea2RegionalMultiLoRAV12",
];
const JSON_WIDGET = "regions_json";

function widgetValue(node, name, fallback) {
  const w = node?.widgets?.find((x) => x.name === name);
  return w === undefined ? fallback : w.value;
}

// NEVER index app.graph.links directly: it is a Map in current frontends
// (1.48.x) and a plain object in older ones, so `links[id]` silently returns
// undefined on the former - which reads exactly like "nothing is wired" and was
// why the chain never reached regions_json. Prefer LiteGraph's own accessor.
function inputLinkedNode(node, inputName) {
  if (!node?.inputs) return null;
  const idx = node.inputs.findIndex((i) => i.name === inputName);
  if (idx < 0) return null;

  if (typeof node.getInputNode === "function") {
    const origin = node.getInputNode(idx);
    if (origin) return origin;
  }

  const linkId = node.inputs[idx].link;
  if (linkId == null) return null;
  const links = app.graph?.links;
  const link = links instanceof Map ? links.get(linkId) : links?.[linkId];
  if (!link) return null;
  return app.graph.getNodeById(link.origin_id) || null;
}

// Walk back up the `characters` chain from the regional node. Chain order is
// box order, so the deepest ancestor is character 1 -> unshift as we climb.
function collectChain(regionalNode) {
  const rows = [];
  const seen = new Set();
  let node = inputLinkedNode(regionalNode, "characters");
  while (node && node.type === CHARACTER_NODE && !seen.has(node.id)) {
    seen.add(node.id);

    // Ctrl+B (bypass, mode 4): the node is dropped but its input is rewired to
    // its output, so the chain flows through it. Skip the character, keep
    // climbing.
    if (node.mode === 4) {
      node = inputLinkedNode(node, "characters");
      continue;
    }
    // Ctrl+M (mute, mode 2): the node AND its links are removed from the
    // prompt, which SEVERS the chain - every character above this one is lost
    // server-side too. Stop here so the rows show what actually arrives.
    if (node.mode === 2) break;

    const hasRef = !!inputLinkedNode(node, "ref_image");
    rows.unshift({
      name: String(widgetValue(node, "name", "") || "").trim(),
      lora: widgetValue(node, "lora_name", "None") || "None",
      strength: Number(widgetValue(node, "strength", 1.1)),
      enable: widgetValue(node, "enable", true) !== false,
      prompt: String(widgetValue(node, "prompt", "") || "").trim(),
      // Left empty on purpose: a wired ref is a live IMAGE tensor with no
      // filename, and writing a placeholder here would both break the row's
      // thumbnail fetch and linger as a dead filename if the chain is later
      // deleted. The server attaches the real tensors at render time; the
      // badge reports how many rows have one.
      ref_image: "",
      ref_enable: widgetValue(node, "ref_enable", true) !== false,
      portrait: widgetValue(node, "portrait", false) === true,
      __ref: hasRef,
    });
    node = inputLinkedNode(node, "characters");
  }
  rows.forEach((r, i) => {
    if (!r.name) r.name = `character${i + 1}`;
  });
  return rows;
}

function syncFromChain(regionalNode) {
  const w = regionalNode.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return;

  const slot = regionalNode.inputs?.find((i) => i.name === "characters");
  const wired = slot && slot.link != null;
  const rows = wired ? collectChain(regionalNode) : [];

  // Nothing wired, or a wired chain that contributes no characters (the nearest
  // one muted, so the link is severed in the prompt). Either way the server
  // falls back to regions_json, so the rows must go back to being the user's
  // own - leaving the last chain on screen would advertise LoRAs that are not
  // going to be applied.
  if (!rows.length) {
    regionalNode.__k2charCount = 0;
    regionalNode.__k2charRefs = 0;
    regionalNode.__k2charActive = 0;
    if (regionalNode.__k2charBackup != null) {
      w.value = regionalNode.__k2charBackup;
      if (w.inputEl) w.inputEl.value = w.value;
      regionalNode.__k2charBackup = null;
      rebuildRegionRows(regionalNode, false);
      regionalNode.setDirtyCanvas(true, true);
    }
    return;
  }

  regionalNode.__k2charCount = rows.length;
  regionalNode.__k2charRefs = rows.filter((r) => r.__ref).length;
  regionalNode.__k2charActive = rows.filter((r) => r.enable).length;

  const next = JSON.stringify(rows.map(({ __ref, ...r }) => r), null, 2);
  // Compare against the widget's LIVE value, not just the last chain we wrote:
  // the box-count auto-sync rewrites these rows behind our back whenever the
  // builder gains or loses a box, and the chain has to win that race.
  if (next === w.value) {
    rebuildRegionRows(regionalNode, true);   // keep the rows locked on redraw
    return;
  }

  if (regionalNode.__k2charBackup == null) regionalNode.__k2charBackup = w.value;
  w.value = next;
  if (w.inputEl) w.inputEl.value = next;
  rebuildRegionRows(regionalNode, true);
  regionalNode.setDirtyCanvas(true, true);
}

// The per-version row UI owns its own widgets; nudge whichever one is present.
// It rebuilds straight from regions_json, so once the chain has written that the
// per-region LoRA combos and strengths already show the chain's values.
function rebuildRegionRows(node, chainDriven) {
  if (typeof node.__k2rebuildRows === "function") {
    try { node.__k2rebuildRows(); } catch (e) { /* row UI absent; JSON still set */ }
  }
  if (!node.widgets) return;
  // Grey the rows out while the chain owns them: they are a live readout, and a
  // manual edit here would be silently overwritten on the next redraw.
  for (const w of node.widgets) {
    if (w.__k2region || w.__k2add) w.disabled = !!chainDriven;
  }
}

app.registerExtension({
  name: "krea2.Character",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!REGIONAL_NODES.includes(nodeData.name)) return;

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (onDrawForeground) onDrawForeground.apply(this, arguments);
      if (this.flags?.collapsed) return;

      // Polling here rather than only on connection events: character widgets
      // change without firing any graph event on the regional node.
      syncFromChain(this);

      const slot = this.inputs?.find((i) => i.name === "characters");
      if (!slot || slot.link == null) return;
      const n = this.__k2charCount || 0;
      const active = this.__k2charActive || 0;
      const refs = this.__k2charRefs || 0;
      // Report active separately: a character with its enable toggle off stays
      // in the rows but claims no box, so "2 wired" alone is misleading.
      const badge =
        `${n} wired · ${active} active (${active} box${active === 1 ? "" : "es"})` +
        ` · ${refs} with ref image — rows are the chain, edit the Character nodes`;

      ctx.save();
      ctx.font = "10px sans-serif";
      const w = ctx.measureText(badge).width + 12;
      ctx.fillStyle = "#2b6cb0";
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(6, this.size[1] + 4, w, 16, 4);
        ctx.fill();
      } else {
        ctx.fillRect(6, this.size[1] + 4, w, 16);
      }
      ctx.fillStyle = "#fff";
      ctx.fillText(badge, 12, this.size[1] + 16);
      ctx.restore();
    };
  },
});
