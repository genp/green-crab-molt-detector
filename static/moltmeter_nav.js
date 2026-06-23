const navHtml = `
<nav class="site-nav" data-site-nav>
    <div class="nav-inner">
        <a class="brand" href="/">
            <img src="/static/GreenCrab.png" alt=""> MoltMeter.ai
        </a>
        <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="siteNavLinks" data-nav-toggle>
            <span class="nav-toggle-bars" aria-hidden="true"></span>
            <span class="sr-only">Menu</span>
        </button>
        <div class="nav-links" id="siteNavLinks" data-nav-links>
            <a href="/">Home</a>
            <a href="/demo">Demo</a>
            <a href="/video">Video</a>
            <a href="/image">Image Upload</a>
            <a href="/field-guide">Field Guide</a>
            <a href="/about-page">About</a>
        </div>
    </div>
</nav>`;

document.querySelectorAll('[data-nav-root]').forEach((root) => {
    root.outerHTML = navHtml;
});

document.querySelectorAll('[data-site-nav]').forEach((nav, index) => {
    const toggle = nav.querySelector('[data-nav-toggle]');
    const links = nav.querySelector('[data-nav-links]');
    if (!toggle || !links) return;
    const linksId = `siteNavLinks${index}`;
    links.id = linksId;
    toggle.setAttribute('aria-controls', linksId);

    toggle.addEventListener('click', () => {
        const expanded = toggle.getAttribute('aria-expanded') === 'true';
        toggle.setAttribute('aria-expanded', String(!expanded));
        nav.classList.toggle('nav-open', !expanded);
    });

    links.querySelectorAll('a').forEach((link) => {
        link.addEventListener('click', () => {
            toggle.setAttribute('aria-expanded', 'false');
            nav.classList.remove('nav-open');
        });
    });
});
