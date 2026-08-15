import { chromium } from "playwright";

const shots: Array<[string, string, number]> = [
  ["replay", "#replay&t=420", 3000],
  ["autopsy", "#autopsy", 2500],
  ["wards", "#wards", 2500],
  ["method", "#method", 1500],
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 2 });
const errors: string[] = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

for (const [name, hash, wait] of shots) {
  // A full reload per shot: navigating to a URL that differs only in its hash does not
  // reload, so a shot taken that way would silently capture the previous view.
  await page.goto("http://localhost:5173/about:blank", { waitUntil: "commit" }).catch(() => {});
  await page.goto(`http://localhost:5173/${hash}`, { waitUntil: "networkidle" });
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForTimeout(wait);
  await page.screenshot({ path: `/tmp/shots/${name}.png`, fullPage: name !== "replay" });
  console.log(`shot ${name}`);
}
await browser.close();
if (errors.length) {
  console.error("page errors:");
  for (const e of [...new Set(errors)]) console.error("  " + e);
  process.exit(1);
}
console.log("no page errors");
