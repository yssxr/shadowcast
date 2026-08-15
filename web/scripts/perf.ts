// Measure the real frame rate during playback, rather than asserting it in a comment.
// Run with the dev server up: `npm run perf`.
import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1728, height: 1080 }, deviceScaleFactor: 2 });
await page.goto("http://localhost:5173/#replay&t=75", { waitUntil: "networkidle" });
await page.waitForTimeout(2500);

async function measure(label: string) {
  await page.keyboard.press("Space");
  await page.waitForTimeout(400);
  const r = await page.evaluate(
    () =>
      new Promise<{ fps: number; worst: number }>((resolve) => {
        const frames: number[] = [];
        let last = performance.now();
        const tick = (now: number) => {
          frames.push(now - last);
          last = now;
          if (frames.length < 180) requestAnimationFrame(tick);
          else {
            const mean = frames.reduce((a, b) => a + b, 0) / frames.length;
            resolve({ fps: 1000 / mean, worst: Math.max(...frames) });
          }
        };
        requestAnimationFrame(tick);
      }),
  );
  await page.keyboard.press("Space");
  console.log(`${label.padEnd(26)} ${r.fps.toFixed(1)} fps, worst frame ${r.worst.toFixed(1)} ms`);
}

await measure("everything on");
await page.getByRole("button", { name: "90% region outline" }).click();
await measure("without the outline");
await page.getByRole("button", { name: "Belief clouds" }).click();
await measure("without belief at all");
await browser.close();
