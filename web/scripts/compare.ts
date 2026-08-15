// Shoot one board at the match's most uncertain moment, for looking at.
import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 1100 }, deviceScaleFactor: 2 });
await page.goto("http://localhost:5173/#replay&t=75.5", { waitUntil: "networkidle" });
await page.waitForTimeout(2500);
await page.locator("canvas").nth(1).screenshot({ path: "/tmp/shots/belief.png" });
console.log("shot /tmp/shots/belief.png");
await browser.close();
