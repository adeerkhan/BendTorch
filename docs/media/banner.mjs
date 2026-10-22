// Builds the README banner and the repository's social preview as SVG, in the visual language of bend-ml's and
// llama.bend's banners: warm paper, one monospace family (embedded, so GitHub renders it), violet for Bend.
// The icon is the launch video's: a layer between two others, a forward arrow above and a backward one below.
//   node docs/media/banner.mjs   -> docs/media/banner.svg, docs/media/social.svg
// social.png (1280 x 640, for the repository settings) is social.svg rendered by a browser:
//   brave --headless --screenshot=docs/media/social.png --window-size=1280,640 docs/media/social.svg
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const out = (name, svg) => fs.writeFileSync(path.join(here, name), svg);
const b64 = (f) => fs.readFileSync(path.join(here, "fonts", f)).toString("base64");
const FONT = `@font-face{font-family:"Bend ML Mono";font-weight:400;src:url(data:font/woff2;base64,${b64("BendMLMono-Regular.woff2")}) format("woff2")}
@font-face{font-family:"Bend ML Mono";font-weight:700;src:url(data:font/woff2;base64,${b64("BendMLMono-Bold.woff2")}) format("woff2")}`;

const C = { bg: "#f2eee7", b1: "#a9a59d", b00: "#87847d", b02: "#4d4a44", vio: "#8b83b5", bar: "#cdc7bc" };
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const T = (x, y, s, { size = 15, fill = C.b02, weight = 400, anchor = "start" } = {}) =>
  `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
const svg = (w, h, rx, title, body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(title)}">
<title>${esc(title)}</title>
<style>${FONT}
text{font-family:"Bend ML Mono",ui-monospace,Menlo,"SF Mono",Consolas,"Liberation Mono",monospace}</style>
<rect width="${w}" height="${h}" rx="${rx}" fill="${C.bg}"/>
${body}
</svg>
`;

// the icon centred on (cx, cy): three cells of size sz (the middle one violet) and the two arrows, with the video's
// proportions (step 1.25 sz, arrow 0.78 of the footprint, shaft 0.13 sz, head 0.34 sz long and 0.42 sz high)
function icon(cx, cy, sz) {
  const step = 1.25 * sz, len = 0.78 * (2 * step + sz), th = 0.13 * sz, hl = 0.34 * sz, hh = 0.42 * sz, r = 0.19 * sz;
  let s = "";
  for (let c = 0; c < 3; c++)
    s += `<rect x="${(cx + (c - 1) * step - sz / 2).toFixed(1)}" y="${(cy - sz / 2).toFixed(1)}" width="${sz}" height="${sz}" rx="${r.toFixed(1)}" fill="${c === 1 ? C.vio : C.bar}"/>`;
  for (const dir of [1, -1]) {
    const y = cy - dir * step, a = cx - (dir * len) / 2, b = cx + (dir * len) / 2, hx = b - dir * hl;
    const p = [[a, y - th / 2], [hx, y - th / 2], [hx, y - hh / 2], [b, y], [hx, y + hh / 2], [hx, y + th / 2], [a, y + th / 2]];
    s += `<path d="M${p.map(([x, yy]) => `${x.toFixed(1)} ${yy.toFixed(1)}`).join(" L")} Z" fill="${C.b00}"/>`;
  }
  return s;
}

// the wordmark and Bend's violet block cursor, centred together (the font advance is 0.6 em)
function wordmark(cx, y, fs) {
  const tw = 9 * 0.6 * fs, cw = 0.5 * fs, gap = 0.12 * fs, x0 = cx - (tw + gap + cw) / 2;
  return T(x0, y, "BendTorch", { size: fs, weight: 700 }) +
    `<rect x="${(x0 + tw + gap).toFixed(1)}" y="${(y - 0.78 * fs).toFixed(1)}" width="${cw}" height="${(0.86 * fs).toFixed(1)}" fill="${C.vio}"/>`;
}

const TITLE = "BendTorch: training neural networks in Bend 2. Gradients in the types, laws checked by the kernel, checked against PyTorch step by step.";
const SUB = "training neural networks in Bend 2";
const LINE = "gradients in the types · laws checked by the kernel · checked against PyTorch, step by step";

{
  const W = 1200, H = 360;
  out("banner.svg", svg(W, H, 14, TITLE, icon(W / 2, 91, 22) + wordmark(W / 2, 214, 92) +
    T(W / 2, 270, SUB, { size: 26, fill: C.b00, anchor: "middle" }) + T(W / 2, 310, LINE, { size: 17, fill: C.b1, anchor: "middle" })));
}
{
  const W = 1280, H = 640;
  out("social.svg", svg(W, H, 0, TITLE, icon(W / 2, 168, 40) + wordmark(W / 2, 382, 124) +
    T(W / 2, 452, SUB, { size: 34, fill: C.b00, anchor: "middle" }) + T(W / 2, 504, LINE, { size: 19, fill: C.b1, anchor: "middle" }) +
    T(W / 2, 584, "github.com/nuxyel/BendTorch", { size: 26, fill: C.vio, weight: 700, anchor: "middle" })));
}
console.log("banner ok");
