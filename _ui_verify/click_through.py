from playwright.sync_api import sync_playwright

out = r"C:\Users\Mubeen\Desktop\Khetgap\khetgap\_ui_verify"

def styles(page, selector):
    loc = page.locator(selector).first
    loc.wait_for(state="visible", timeout=20000)
    return loc.evaluate(
        """el => {
            const cs = getComputedStyle(el);
            const text = el.querySelector('p, span, div') || el;
            const ts = getComputedStyle(text);
            return {
                bg: cs.backgroundColor,
                color: cs.color,
                border: cs.borderTopColor + ' ' + cs.borderTopWidth,
                textColor: ts.color,
                label: (el.innerText || '').trim().slice(0, 80)
            };
        }"""
    )

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded", timeout=30000)
    page.get_by_test_id("stBaseButton-primary").wait_for(timeout=30000)

    reset = page.get_by_role("button", name="Reset Analysis")
    run = page.get_by_role("button", name="Run Analysis")
    print("RESET", styles(page, 'button[data-testid="stBaseButton-secondary"]'))
    print("RUN", styles(page, 'button[data-testid="stBaseButton-primary"]'))

    reset.hover()
    page.wait_for_timeout(250)
    print("RESET_HOVER", styles(page, 'button[data-testid="stBaseButton-secondary"]'))
    reset.blur()
    page.wait_for_timeout(150)
    page.screenshot(path=f"{out}\\01_idle.png", full_page=False)

    run.hover()
    page.wait_for_timeout(250)
    print("RUN_HOVER", styles(page, 'button[data-testid="stBaseButton-primary"]'))
    page.screenshot(path=f"{out}\\02_run_hover.png", full_page=False)
    page.mouse.move(0, 0)
    page.wait_for_timeout(150)

    reset.click()
    page.get_by_text("Please select or upload").wait_for(timeout=15000)
    page.screenshot(path=f"{out}\\03_after_reset.png", full_page=False)

    page.locator('[data-testid="stSelectbox"]').first.click()
    page.get_by_role("option", name="sugarcane_hard.png").click()
    page.wait_for_timeout(800)

    page.get_by_role("button", name="Run Analysis").click()
    page.get_by_text("Download HD Overlay").wait_for(timeout=60000)
    print("DOWNLOAD", styles(page, '[data-testid="stDownloadButton"] button'))
    print("METRIC", styles(page, '[data-testid="stMetricLabel"]'))
    print("CAPTION", styles(page, '[data-testid="stImageCaption"]'))
    page.screenshot(path=f"{out}\\04_after_run.png", full_page=False)
    dl_hover = page.locator('[data-testid="stDownloadButton"] button').first
    dl_hover.hover()
    page.wait_for_timeout(200)
    print("DOWNLOAD_HOVER", styles(page, '[data-testid="stDownloadButton"] button'))

    overlay_tab = page.get_by_role("tab", name="Result Overlay")
    if overlay_tab.count():
        overlay_tab.click()
        page.wait_for_timeout(400)
    page.screenshot(path=f"{out}\\05_overlay.png", full_page=True)

    browser.close()
    print("DONE")
