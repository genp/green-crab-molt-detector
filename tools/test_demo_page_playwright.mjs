#!/usr/bin/env node
/**
 * Browser regression test for the MoltMeter demo page.
 *
 * Usage:
 *   node tools/test_demo_page_playwright.mjs [baseUrl]
 *
 * The app should already be running, for example:
 *   /Users/gen/.venv/green_crabs_local/bin/python3 -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8080
 */

import assert from 'node:assert/strict';

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (error) {
  console.error('Playwright is required for this test. Install with: npm install -D playwright');
  console.error('Then install the browser once with: npx playwright install chromium');
  process.exit(2);
}

const baseUrl = process.argv[2] || 'http://127.0.0.1:8080';
const demoUrl = `${baseUrl.replace(/\/$/, '')}/demo`;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

try {
  const requests = [];
  page.on('request', (request) => {
    const url = request.url();
    if (url.includes('/static/demo_videos/') || url.includes('/predict_stream')) {
      requests.push(url);
    }
  });

  await page.goto(demoUrl, { waitUntil: 'domcontentloaded' });

  await page.waitForSelector('#demoVideo');
  await page.waitForSelector('#thumbnailStrip button');

  const clipCount = await page.locator('#thumbnailStrip button').count();
  assert.ok(clipCount >= 2, `expected at least 2 demo clips, got ${clipCount}`);

  await page.waitForFunction(() => {
    const video = document.querySelector('#demoVideo');
    return video && video.currentSrc && video.readyState >= HTMLMediaElement.HAVE_METADATA;
  });

  const firstSrc = await page.locator('#demoVideo').evaluate((video) => video.currentSrc);
  assert.match(firstSrc, /\/static\/demo_videos\/demo_\d+\.mp4/, 'video should load a demo mp4');

  await page.locator('#demoVideo').evaluate((video) => video.play());
  await page.waitForFunction(() => {
    const video = document.querySelector('#demoVideo');
    return video && !video.paused && video.currentTime > 0;
  }, { timeout: 10000 });

  await page.waitForFunction(() => {
    const days = document.querySelector('#demoDays')?.textContent || '';
    const phase = document.querySelector('#demoPhase')?.textContent || '';
    return days !== 'N/A' && phase !== 'N/A';
  }, { timeout: 10000 });

  const overlayHasDrawing = await page.locator('#demoOverlay').evaluate((canvas) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || canvas.width === 0 || canvas.height === 0) return false;
    const sample = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let index = 3; index < sample.length; index += 4) {
      if (sample[index] !== 0) return true;
    }
    return false;
  });
  assert.equal(overlayHasDrawing, true, 'overlay canvas should contain detection drawing');

  const timelineLoaded = requests.some((url) => url.endsWith('_timeline.json'));
  const liveFallbackUsed = requests.some((url) => url.includes('/predict_stream'));
  assert.ok(timelineLoaded || liveFallbackUsed, 'demo should use a timeline or live /predict_stream fallback');

  await page.locator('#demoVideo').evaluate((video) => {
    video.currentTime = Math.max(0, video.duration - 0.15);
  });
  await page.waitForFunction((oldSrc) => {
    const video = document.querySelector('#demoVideo');
    return video && video.currentSrc !== oldSrc;
  }, firstSrc, { timeout: 10000 });
  const rotatedSrc = await page.locator('#demoVideo').evaluate((video) => video.currentSrc);
  assert.notEqual(rotatedSrc, firstSrc, 'auto-rotate should advance to the next video');

  const secondButton = page.locator('#thumbnailStrip button').nth(1);
  await secondButton.click();
  const selectedSrc = await page.locator('#demoVideo').evaluate((video) => video.currentSrc);
  await page.locator('#demoVideo').evaluate((video) => {
    video.currentTime = Math.max(0, video.duration - 0.15);
  });
  await page.waitForTimeout(1200);
  const repeatedSrc = await page.locator('#demoVideo').evaluate((video) => video.currentSrc);
  assert.equal(repeatedSrc, selectedSrc, 'manual selection should repeat the selected clip');

  await page.locator('#resumeRotationBtn').click();
  await page.locator('#demoVideo').evaluate((video) => {
    video.currentTime = Math.max(0, video.duration - 0.15);
  });
  await page.waitForFunction((oldSrc) => {
    const video = document.querySelector('#demoVideo');
    return video && video.currentSrc !== oldSrc;
  }, selectedSrc, { timeout: 10000 });

  console.log('Demo page browser test passed');
} finally {
  await browser.close();
}
