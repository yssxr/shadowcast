// Measure the real frame rate during playback, rather than asserting it in a comment.
import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1728, height: 1080 }, deviceScaleFactor: 2 });
await page.goto("http://localhost:5173/#replay&t=300", { waitUntil: "networkidle" });
await page.waitForTimeout(2500);

for (const mode of ["Cloud", "Contour", "Grid"]) {
  await page.getByRole("button", { name: mode, exact: true }).click();
  await page.keyboard.press("Space");           // play
  await page.waitForTimeout(400);               // let it settle
  const fps = await page.evaluate(
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
  await page.keyboard.press("Space");           // pause
  console.log(
    `${mode.padEnd(8)} ${fps.fps.toFixed(1)} fps mean, worst frame ${fps.worst.toFixed(1)} ms`,
  );
}
await browser.close();
