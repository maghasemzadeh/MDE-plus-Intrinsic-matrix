/* Thesis defense deck engine — no dependencies except KaTeX (bundled, optional).
   Modes:  (default) audience view   ?presenter=1 presenter view   ?embed=1 preview frame
           ?remote=1 phone presenter view, driven through serve.py (see README) */
(function () {
  'use strict';

  var qs = new URLSearchParams(location.search);
  var REMOTE = qs.has('remote');
  var MODE = (qs.has('presenter') || REMOTE) ? 'presenter' : (qs.has('embed') ? 'embed' : 'main');
  document.body.classList.add(MODE);
  if (REMOTE) document.body.classList.add('remote');
  /* served by serve.py: the audience window and the phone talk through the server (POST up, Server-Sent Events down) */
  var SERVED = /^https?:$/.test(location.protocol), TOKEN = qs.get('k') || '';
  var DECK_V = 3;   /* raised when the phone and the audience window must run the same engine; a mismatch tells the phone to ask for a reload */
  function api(path) { return '/api/' + path + (TOKEN ? '?k=' + encodeURIComponent(TOKEN) : ''); }
  function post(path, body) {
    try { fetch(api(path), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), keepalive: true }).catch(function () {}); } catch (e) {}
  }

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var fa = function (v) { return String(v).replace(/[0-9]/g, function (d) { return '۰۱۲۳۴۵۶۷۸۹'[d]; }).replace(/\./g, '٫'); };
  var SVG = 'http://www.w3.org/2000/svg';
  function el(name, attrs, parent, text) {
    var e = document.createElementNS(SVG, name);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  var CFG = JSON.parse($('#deck-config').textContent);
  var stage = $('#stage');
  var slides = $$('.slide', stage);
  var mainCount = slides.filter(function (s) { return !s.classList.contains('backup'); }).length;
  var backupCount = slides.length - mainCount;
  var steps = slides.map(function (s) {
    var m = 0;
    $$('.frag', s).forEach(function (f) { m = Math.max(m, +f.getAttribute('data-f') || 1); });
    return m;
  });
  /* .auto items enter by themselves when their slide opens; --i sets each one's place in the sequence */
  slides.forEach(function (s) { $$('.auto', s).forEach(function (a, n) { a.style.setProperty('--i', n); }); });
  var state = { i: 0, step: 0 };

  /* ---------------- maths ---------------- */
  function renderMath() {
    if (!window.katex) return;
    $$('.tex').forEach(function (n) {
      try {
        window.katex.render(n.getAttribute('data-tex'), n, { displayMode: n.classList.contains('block'), throwOnError: false, trust: true, strict: false });
      } catch (e) { n.textContent = n.getAttribute('data-tex'); }
    });
  }

  /* ---------------- charts ---------------- */
  function colorOf(key) { return 'var(--c-' + key + ')'; }
  function fragAttrs(g, f) { if (f) { g.classList.add('frag'); g.setAttribute('data-f', f); } }
  function fmt(v, dec) { return (+v).toFixed(dec === undefined ? 3 : dec); }

  function yAxis(svg, spec, x0, x1, y0, y1, sy) {
    var grid = el('g', { 'class': 'grid' }, svg);
    (spec.yticks || []).forEach(function (t) {
      var y = sy(t);
      el('line', { x1: x0, x2: x1, y1: y, y2: y }, grid);
      el('text', { 'class': 'tick', x: x0 - 14, y: y + 8, 'text-anchor': 'end' }, svg, fmt(t, spec.tickDec === undefined ? 2 : spec.tickDec));
    });
    var ax = el('g', { 'class': 'axis' }, svg);
    el('line', { x1: x0, x2: x1, y1: y0, y2: y0 }, ax);
    if (spec.ylabel) {
      var t = el('text', { 'class': 'ylab', x: 0, y: 0, 'text-anchor': 'middle', direction: 'rtl', transform: 'translate(' + (x0 - 92) + ',' + ((y0 + y1) / 2) + ') rotate(-90)' }, svg, spec.ylabel);
      t.style.unicodeBidi = 'plaintext';
    }
  }

  function barPath(x, y, w, yBase, r) {
    r = Math.min(r, w / 2, Math.max(0, yBase - y));
    return 'M' + x + ',' + yBase + 'V' + (y + r) + 'Q' + x + ',' + y + ' ' + (x + r) + ',' + y + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + 'V' + yBase + 'Z';
  }

  function chartBars(box, spec) {
    var W = spec.w || 1500, H = spec.h || 640;
    var m = { l: spec.ml || 130, r: spec.mr || 30, t: spec.mt || 50, b: spec.mb || 96 };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, box);
    var x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
    var sy = function (v) { return y0 - (v / spec.ymax) * (y0 - y1); };
    yAxis(svg, spec, x0, x1, y0, y1, sy);
    var ng = spec.groups.length, ns = spec.series.length;
    var band = (x1 - x0) / ng;
    var gap = spec.gap === undefined ? 8 : spec.gap;
    var bw = Math.min(spec.barW || 78, (band * 0.78 - gap * (ns - 1)) / ns);
    var groupW = bw * ns + gap * (ns - 1);
    spec.groups.forEach(function (g, gi) {
      var gx = x0 + band * gi + (band - groupW) / 2;
      var cx = x0 + band * (gi + 0.5);
      var lab = el('text', { 'class': 'cat', x: cx, y: y0 + 44, 'text-anchor': 'middle' }, svg, g.label);
      if (g.rtl) { lab.setAttribute('direction', 'rtl'); lab.style.unicodeBidi = 'plaintext'; }
      if (g.sub) {
        var sb = el('text', { 'class': 'cat sub', x: cx, y: y0 + 76, 'text-anchor': 'middle', direction: 'rtl' }, svg, g.sub);
        sb.style.unicodeBidi = 'plaintext';
      }
      spec.series.forEach(function (s, si) {
        var v = s.values[gi];
        if (v === null || v === undefined) return;
        var bx = gx + si * (bw + gap), by = sy(v);
        var grp = el('g', { 'class': 'bar-g' }, svg);
        fragAttrs(grp, (s.fs && s.fs[gi]) || s.f);
        var col = (s.colors && s.colors[gi]) || colorOf(s.key);
        var p = el('path', { 'class': 'bar', d: barPath(bx, by, bw, y0, 7), fill: col }, grp);
        p.style.transitionDelay = (gi * 90 + si * 60) + 'ms';
        el('title', {}, p, (s.name || s.key) + ' — ' + g.label + ': ' + fmt(v, spec.dec));
        var hero = s.hero || (s.heroes && s.heroes[gi]);
        el('text', { 'class': 'val' + (hero ? ' hero' : ''), x: bx + bw / 2, y: by - 12, 'text-anchor': 'middle' }, grp, fmt(v, spec.dec));
      });
      if (g.callout) {
        var cg = el('g', { 'class': 'co' }, svg);
        fragAttrs(cg, g.calloutF);
        el('text', { 'class': 'callout', x: cx, y: y1 - 8, 'text-anchor': 'middle' }, cg, g.callout);
      }
    });
    (spec.marks || []).forEach(function (mk) {
      var gx = x0 + band * mk.g + (band - groupW) / 2 + mk.s * (bw + gap) + bw / 2, y = sy(mk.value);
      var g = el('g', { 'class': 'mk' }, svg);
      fragAttrs(g, mk.f);
      el('line', { x1: gx - bw / 2 - 8, x2: gx + bw / 2 + 8, y1: y, y2: y, stroke: 'var(--navy-950)', 'stroke-width': 5, 'stroke-dasharray': '10 7' }, g);
      el('text', { 'class': 'val', x: gx + bw / 2 + 16, y: y + 8, 'text-anchor': 'start' }, g, fmt(mk.value, spec.dec));
    });
    (spec.refs || []).forEach(function (rf) {
      var y = sy(rf.value);
      el('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: 'var(--ink-3)', 'stroke-width': 2.5, 'stroke-dasharray': '6 8' }, svg);
    });
  }

  function chartLine(box, spec) {
    var W = spec.w || 1000, H = spec.h || 760;
    var m = { l: 120, r: 40, t: 30, b: 110 };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, box);
    var x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
    var tr = function (v) { return spec.log ? Math.log(v) : v; };
    var sx = function (v) { return x0 + (tr(v) - tr(spec.min)) / (tr(spec.max) - tr(spec.min)) * (x1 - x0); };
    var sy = function (v) { return y0 - (tr(v) - tr(spec.min)) / (tr(spec.max) - tr(spec.min)) * (y0 - y1); };
    var grid = el('g', { 'class': 'grid' }, svg);
    spec.ticks.forEach(function (t) {
      el('line', { x1: x0, x2: x1, y1: sy(t), y2: sy(t) }, grid);
      el('line', { x1: sx(t), x2: sx(t), y1: y0, y2: y1 }, grid);
      el('text', { 'class': 'tick', x: x0 - 14, y: sy(t) + 8, 'text-anchor': 'end' }, svg, t);
      el('text', { 'class': 'tick', x: sx(t), y: y0 + 36, 'text-anchor': 'middle' }, svg, t);
    });
    var ax = el('g', { 'class': 'axis' }, svg);
    el('line', { x1: x0, x2: x1, y1: y0, y2: y0 }, ax);
    el('line', { x1: x0, x2: x0, y1: y0, y2: y1 }, ax);
    var xl = el('text', { 'class': 'ylab', x: (x0 + x1) / 2, y: H - 22, 'text-anchor': 'middle', direction: 'rtl' }, svg, spec.xlabel);
    xl.style.unicodeBidi = 'plaintext';
    var yl = el('text', { 'class': 'ylab', 'text-anchor': 'middle', direction: 'rtl', transform: 'translate(34,' + ((y0 + y1) / 2) + ') rotate(-90)' }, svg, spec.ylabel);
    yl.style.unicodeBidi = 'plaintext';
    spec.series.forEach(function (s) {
      var g = el('g', { 'class': 'ln-g' }, svg);
      fragAttrs(g, s.f);
      var d = s.points.map(function (p, i) { return (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1); }).join('');
      var path = el('path', { 'class': 'ln' + (s.dash ? ' dash' : ''), d: d, stroke: s.color || colorOf(s.key) }, g);
      if (!s.dash) {
        var len = Math.ceil(path.getTotalLength ? path.getTotalLength() : 2000) + 4;
        path.style.setProperty('--len', len);
        s.points.forEach(function (p) {
          var c = el('circle', { 'class': 'dot', cx: sx(p[0]), cy: sy(p[1]), r: 11, fill: s.color || colorOf(s.key) }, g);
          el('title', {}, c, p[0] + ' → ' + p[1]);
        });
      }
    });
  }

  function chartDumbbell(box, spec) {
    var W = spec.w || 1400, rowH = spec.rowH || 118;
    var m = { l: 250, r: 190, t: 20, b: 70 };
    var H = m.t + m.b + rowH * spec.rows.length;
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, box);
    var x0 = m.l, x1 = W - m.r, yb = H - m.b;
    var sx = function (v) { return x0 + (v - spec.min) / (spec.max - spec.min) * (x1 - x0); };
    var grid = el('g', { 'class': 'grid' }, svg);
    spec.ticks.forEach(function (t) {
      el('line', { x1: sx(t), x2: sx(t), y1: m.t, y2: yb }, grid);
      el('text', { 'class': 'tick', x: sx(t), y: yb + 34, 'text-anchor': 'middle' }, svg, fmt(t, 2));
    });
    spec.rows.forEach(function (r, i) {
      var y = m.t + rowH * (i + 0.5);
      el('text', { 'class': 'cat', x: x0 - 30, y: y + 9, 'text-anchor': 'end' }, svg, r.label);
      var g = el('g', { 'class': 'db' }, svg);
      fragAttrs(g, r.f);
      var ln = el('line', { 'class': 'link', x1: sx(r.b), x2: sx(r.a), y1: y, y2: y }, g);
      ln.style.transitionDelay = (i * 120) + 'ms';
      el('circle', { 'class': 'dot', cx: sx(r.a), cy: y, r: 17, fill: colorOf('control') }, g);
      el('circle', { 'class': 'dot', cx: sx(r.b), cy: y, r: 17, fill: colorOf('proposed') }, g);
      el('text', { 'class': 'val', x: sx(r.a) + 28, y: y + 8, 'text-anchor': 'start' }, g, fmt(r.a));
      el('text', { 'class': 'val hero', x: sx(r.b) - 28, y: y + 9, 'text-anchor': 'end' }, g, fmt(r.b));
      el('text', { 'class': 'callout', x: W - 14, y: y + 11, 'text-anchor': 'end' }, g, r.pct);
    });
  }

  function renderCharts() {
    $$('.chart[data-chart]').forEach(function (box) {
      var spec = JSON.parse($('script', box).textContent);
      var kind = box.getAttribute('data-chart');
      if (kind === 'bars') chartBars(box, spec);
      else if (kind === 'line') chartLine(box, spec);
      else if (kind === 'dumbbell') chartDumbbell(box, spec);
    });
    $$('[data-heat]').forEach(function (td) {
      var v = parseFloat(td.getAttribute('data-heat'));
      var a = Math.min(1, Math.abs(v) / 50);
      if (v < 0) { td.style.background = 'rgba(24,79,149,' + (0.10 + 0.78 * a).toFixed(2) + ')'; td.style.color = a > 0.42 ? '#fff' : 'var(--navy-900)'; }
      else if (v > 0) { td.style.background = 'rgba(201,98,31,' + (0.10 + 0.78 * a).toFixed(2) + ')'; }
    });
  }

  /* ---------------- chrome ---------------- */
  function buildChrome() {
    var nav = $('#partnav');
    CFG.parts.forEach(function (p, i) {
      var d = document.createElement('div');
      d.className = 'p'; d.setAttribute('data-part', i + 1);
      d.innerHTML = '<span class="n">' + fa(i + 1) + '</span><span>' + p + '</span>';
      nav.appendChild(d);
      if (MODE === 'main') d.addEventListener('click', function () {
        for (var k = 0; k < mainCount; k++) if (+slides[k].getAttribute('data-part') === i + 1) { command({ cmd: 'goto', i: k, step: 0 }); return; }
      });
    });
    var bk = document.createElement('div'); bk.className = 'bk'; bk.textContent = CFG.backupLabel; nav.appendChild(bk);
  }

  function updateChrome() {
    var s = slides[state.i];
    var part = +s.getAttribute('data-part') || 0;
    var isBackup = s.classList.contains('backup');
    $('#chrome').classList.toggle('hidden', s.hasAttribute('data-nochrome'));
    $('#partnav').classList.toggle('backup', isBackup);
    $$('#partnav .p').forEach(function (p) {
      var n = +p.getAttribute('data-part');
      p.classList.toggle('cur', n === part); p.classList.toggle('done', n < part);
    });
    var c = $('#counter');
    if (isBackup) c.innerHTML = CFG.backupLabel + ' ' + fa(state.i - mainCount + 1) + ' <span class="tot">' + CFG.of + ' ' + fa(backupCount) + '</span>';
    else c.innerHTML = fa(state.i + 1) + ' <span class="tot">' + CFG.of + ' ' + fa(mainCount) + '</span>';
    $('#progress i').style.width = (isBackup ? 100 : ((state.i + 1) / mainCount * 100)) + '%';
  }

  /* ---------------- navigation ---------------- */
  function applyFrags(s, step) {
    $$('.frag', s).forEach(function (f) {
      var n = +f.getAttribute('data-f') || 1;
      f.classList.toggle('on', n <= step);
      f.classList.toggle('past', n < step);
    });
    s.setAttribute('data-step', step);
  }

  /* ---------------- video ---------------- */
  function activeVideo() { return $('video:not([data-ambient])', slides[state.i]); }
  function playVideo(v) {
    /* if the browser refuses to start it with sound, start it silent; a play() cut short by pause() is not retried */
    try {
      var p = v.play();
      if (p && p.catch) p.catch(function (err) {
        if (!err || err.name !== 'NotAllowedError' || v.muted) return;
        v.muted = true; var q = v.play(); if (q && q.catch) q.catch(function () {});
      });
    } catch (e) {}
  }
  /* a video with data-play="N" starts on step N of its slide. `from` is the step we came from on the same slide, or null when the slide was just entered */
  function syncVideos(s, step, from) {
    if (MODE !== 'main') return;
    $$('video[data-play]', s).forEach(function (v) {
      var n = +v.getAttribute('data-play') || 1;
      if (from === null || step < n) { v.pause(); try { v.currentTime = 0; } catch (e) {} }
      else if (from < n && v.paused) { if (v.ended) v.currentTime = 0; playVideo(v); }
    });
    /* a video with data-ambient plays by itself, silent and in a loop, whenever its slide is shown; it is paused on leaving like any other video, and the video panel and keys ignore it */
    $$('video[data-ambient]', s).forEach(function (v) { v.muted = true; v.loop = true; if (v.paused) playVideo(v); });
  }
  function videoState() {
    var v = activeVideo();
    return v ? { t: v.currentTime || 0, d: v.duration || 0, paused: v.paused, muted: v.muted } : null;
  }
  function videoCmd(c) {
    var v = activeVideo();
    if (!v || MODE !== 'main') return;
    var d = v.duration || 0;
    if (c.act === 'toggle') { if (v.paused) { if (v.ended) v.currentTime = 0; playVideo(v); } else v.pause(); }
    else if (c.act === 'restart') { v.currentTime = 0; playVideo(v); }
    else if (c.act === 'nudge') v.currentTime = Math.max(0, Math.min(d, v.currentTime + c.d));
    else if (c.act === 'seek') v.currentTime = Math.max(0, Math.min(d, +c.t || 0));
    else if (c.act === 'mute') v.muted = !v.muted;
    broadcastVideo();
  }
  /* preview frame in the presenter window: follow the audience window's video, always silent */
  function mirrorVideo(st) {
    var v = activeVideo();
    if (!v || !st) return;
    v.muted = true;
    try { if (Math.abs(v.currentTime - st.t) > 0.4) v.currentTime = st.t; } catch (e) {}
    if (st.paused) v.pause(); else if (v.paused) playVideo(v);
  }

  function show(i, step, dir, silent) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    step = Math.max(0, Math.min(steps[i], step));
    var prev = slides[state.i], next = slides[i], from = prev === next ? state.step : null;
    if (prev !== next) {
      stage.style.setProperty('--dir', dir >= 0 ? 1 : -1);
      prev.classList.remove('active'); prev.classList.add('leaving');
      setTimeout(function () { prev.classList.remove('leaving'); }, 500);
      $$('video', prev).forEach(function (v) { v.pause(); });
      next.classList.toggle('back', dir < 0);
      next.classList.add('active');
    } else next.classList.add('active');
    state.i = i; state.step = step;
    applyFrags(next, step);
    syncVideos(next, step, from);
    updateChrome();
    if (MODE === 'main') { try { history.replaceState(null, '', '#' + (i + 1)); } catch (e) { location.hash = i + 1; } }
    if (!silent) broadcast();
  }
  function next() {
    if (state.step < steps[state.i]) show(state.i, state.step + 1, 1);
    else if (state.i < slides.length - 1) show(state.i + 1, 0, 1);
  }
  function prev() {
    if (state.step > 0) show(state.i, state.step - 1, -1);
    else if (state.i > 0) show(state.i - 1, steps[state.i - 1], -1);
  }

  function fit() {
    var w = window.innerWidth, h = window.innerHeight;
    var k = Math.min(w / 1920, h / 1080);
    stage.style.transform = 'translate(' + (-960 * k) + 'px,' + (-540 * k) + 'px) scale(' + k + ')';
  }

  /* ---------------- overview ---------------- */
  function buildOverview() {
    var ol = $('#overview ol'), lastPart = -1;
    slides.forEach(function (s, i) {
      var part = s.classList.contains('backup') ? 99 : (+s.getAttribute('data-part') || 0);
      if (part !== lastPart && part > 0) {
        var h = document.createElement('li'); h.className = 'parthead';
        h.textContent = part === 99 ? CFG.backupLabel : fa(part) + '. ' + CFG.parts[part - 1];
        ol.appendChild(h); lastPart = part;
      }
      var li = document.createElement('li');
      var num = s.classList.contains('backup') ? '+' + fa(i - mainCount + 1) : fa(i + 1);
      li.innerHTML = '<b>' + num + '</b><span></span>';
      li.lastChild.textContent = s.getAttribute('data-title') || '';
      li.setAttribute('data-i', i);
      li.addEventListener('click', function () { toggleOverview(false); command({ cmd: 'goto', i: i, step: 0 }); });
      ol.appendChild(li);
    });
  }
  function toggleOverview(force) {
    var o = $('#overview');
    var open = force === undefined ? !o.classList.contains('open') : force;
    o.classList.toggle('open', open);
    $$('li[data-i]', o).forEach(function (li) { li.classList.toggle('cur', +li.getAttribute('data-i') === state.i); });
    var cur = $('li.cur', o);
    if (open && cur && cur.scrollIntoView) cur.scrollIntoView({ block: 'center' });
  }

  /* ---------------- sync between windows ---------------- */
  var presenterWin = null;
  function broadcast() {
    var msg = { type: 'deck-state', i: state.i, step: state.step, video: videoState() };
    if (MODE === 'main' && presenterWin && !presenterWin.closed) presenterWin.postMessage(msg, '*');
    if (MODE === 'main' && server.on) post('state', { i: state.i, step: state.step, video: msg.video, v: DECK_V });
    if (MODE === 'presenter') updatePresenter();
  }
  var vidTimer = null, vidLast = 0;
  function broadcastVideo(ev) {
    if (MODE === 'main' && presenterWin && !presenterWin.closed) presenterWin.postMessage({ type: 'deck-video', video: videoState() }, '*');
    if (MODE !== 'main' || !server.on) return;
    /* timeupdate fires several times a second: send at most two a second to the phone, other events at once */
    var send = function () { vidTimer = null; vidLast = Date.now(); post('video', { video: videoState() }); };
    if (ev && ev.type === 'timeupdate') { if (!vidTimer) vidTimer = setTimeout(send, Math.max(0, 500 - (Date.now() - vidLast))); }
    else { clearTimeout(vidTimer); send(); }
  }
  function command(c) {
    if (REMOTE) { post('cmd', { c: c }); return; }
    if (MODE === 'presenter' && window.opener && !window.opener.closed) { window.opener.postMessage({ type: 'deck-cmd', c: c }, '*'); return; }
    if (c.cmd === 'next') next(); else if (c.cmd === 'prev') prev(); else if (c.cmd === 'goto') show(c.i, c.step || 0, c.i >= state.i ? 1 : -1);
    else if (c.cmd === 'video') videoCmd(c);
    else if (c.cmd === 'pointer') showLaser(c);
  }
  /* pointer from the phone: x and y are fractions of the slide; `n` orders the messages, so a late one cannot pull the dot back */
  var laser = null, laserN = 0, laserT = null;
  function showLaser(c) {
    if (MODE !== 'main') return;
    if (!laser) { laser = document.createElement('div'); laser.id = 'laser'; stage.appendChild(laser); }
    if (c.n && c.n < laserN && laserN - c.n < 400) return;   /* n is the sender's clock in ms, so it keeps growing across reloads */
    laserN = c.n || 0;
    clearTimeout(laserT);
    if (c.off) { laser.classList.remove('on'); return; }
    var fresh = !laser.classList.contains('on');   /* a new touch appears in place instead of gliding over from the last one */
    if (fresh) laser.style.transition = 'none';
    laser.style.transform = 'translate(' + (c.x * 1920).toFixed(1) + 'px,' + (c.y * 1080).toFixed(1) + 'px)';
    if (fresh) { void laser.offsetWidth; laser.style.transition = ''; }
    laser.classList.add('on');
    laserT = setTimeout(function () { laser.classList.remove('on'); }, 4000);   /* a lost "finger up" must not leave the dot behind */
  }
  window.addEventListener('message', function (ev) { onMsg(ev.data || {}, ev.source); });
  function onMsg(d, source) {
    if (d.type === 'deck-state') { if (MODE === 'presenter') lastVideo = d.video || null; show(d.i, d.step, d.i >= state.i ? 1 : -1, MODE !== 'presenter'); if (MODE === 'presenter') updatePresenter(); }
    else if (d.type === 'deck-cmd' && MODE === 'main') command(d.c);
    else if (d.type === 'deck-hello' && MODE === 'main') { presenterWin = source; broadcast(); }
    else if (d.type === 'deck-ready' && MODE === 'presenter') updatePresenter();
    else if (d.type === 'deck-key' && MODE === 'presenter') onKey({ key: d.key, code: d.code, preventDefault: function () {} });
    else if (d.type === 'deck-video' && MODE === 'presenter') { lastVideo = d.video || null; updateVideoPanel(); if (frames[0] && frames[0].contentWindow) frames[0].contentWindow.postMessage(d, '*'); }
    else if (d.type === 'deck-video' && MODE === 'embed') mirrorVideo(d.video);
    else if (d.type === 'deck-notes') { setNoteText(d.key, d.kind, d.text); if (MODE === 'presenter') renderNotes(slides[state.i]); }
  }

  /* ---------------- link through serve.py ---------------- */
  var server = { on: false, es: null, mains: -1 };
  function connectServer() {
    if (!SERVED || (MODE !== 'main' && !REMOTE)) return;
    fetch(api('ping'), { cache: 'no-store' }).then(function (r) { return r.status === 403 ? { denied: true } : r.ok ? r.json() : null; }).then(function (j) {
      if (j && j.denied) { setConn('denied'); return; }
      if (!j || !j.deck) return;   /* a plain static server: stay a local deck */
      server.on = true; openEvents();
    }).catch(function () { setConn('bad'); setTimeout(connectServer, 3000); });
  }
  function openEvents() {
    if (server.es) server.es.close();
    var es = server.es = new EventSource(api('events') + (TOKEN ? '&' : '?') + 'role=' + (REMOTE ? 'remote' : 'main'));
    es.onopen = function () { if (MODE === 'main') broadcast(); };
    es.onerror = function () { setConn('bad'); };
    es.addEventListener('cmd', function (e) { if (MODE === 'main') command(JSON.parse(e.data).c); });
    es.addEventListener('state', function (e) {
      var d = JSON.parse(e.data); d.type = 'deck-state'; onMsg(d);
      if (REMOTE) { server.stale = d.v !== DECK_V; setConn(server.mains > 0 ? 'ok' : 'nomain'); }
    });
    es.addEventListener('video', function (e) { var d = JSON.parse(e.data); d.type = 'deck-video'; onMsg(d); });
    es.addEventListener('notes', function (e) { var d = JSON.parse(e.data); d.type = 'deck-notes'; onMsg(d); });
    es.addEventListener('status', function (e) { server.mains = JSON.parse(e.data).mains; setConn(server.mains > 0 ? 'ok' : 'nomain'); });
  }
  function setConn(st) {
    var c = $('#pv-conn');
    if (!REMOTE || !c) return;
    if (st === 'ok' && server.stale) st = 'stale';   /* the audience window still runs an older engine: pointer and other new commands would be ignored */
    c.className = 'conn ' + st;
    c.textContent = CFG.conn[st] || '';
    document.body.classList.toggle('offline', st !== 'ok');
  }
  /* a phone that slept may have lost the stream; reconnect when the page is shown again */
  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState !== 'visible' || !SERVED) return;
    if (server.on && (!server.es || server.es.readyState === 2)) openEvents();
    if (REMOTE) { lock = null; keepAwake(); }
  });

  /* ---------------- presenter view ---------------- */
  var t0 = null, frames = [], lastVideo = null, seeking = false;
  function clock(s) { s = Math.max(0, Math.floor(s || 0)); return fa(Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0')); }
  function updateVideoPanel() {
    var box = $('#pv-video'), has = !!activeVideo();
    box.hidden = !has;
    if (!has) return;
    var st = lastVideo || { t: 0, d: 0, paused: true, muted: true };
    box.classList.toggle('playing', !st.paused); box.classList.toggle('sound', !st.muted);
    if (!seeking) $('#pv-vseek').value = st.d ? Math.round(st.t / st.d * 1000) : 0;
    $('#pv-vtime').textContent = clock(st.t) + ' / ' + clock(st.d);
  }
  /* notes are plain text. Inline: an English term in parentheses stays on one line in a lighter colour, and
     Persian-digit numbers are highlighted. Full notes are broken into short paragraphs; a summary (one item
     per line: "1. " numbered, "- " bullet, "! " caveat) is rendered as lists. */
  var EN_RE = /(\([A-Za-z0-9][A-Za-z0-9 \-,]*\))/, NUM_RE = /([\u2212+\-]?[\u06F0-\u06F9][\u06F0-\u06F9\u066B\u066C]*(?:%|\u066A)?)/;
  /* a Latin name such as DA3-Metric or Depth Anything 3 is kept whole and isolated, so the RTL line cannot split it */
  var LAT_RE = /([A-Za-z][A-Za-z0-9\-.\/×]*(?: [A-Za-z0-9][A-Za-z0-9\-.\/×]*)*)/;
  /* $...$ in notes is typeset with KaTeX as an isolated left-to-right formula, so it cannot be scrambled by the RTL line */
  var TEX_RE = /(\$[^$]+\$)/;
  function inline(el, text) {
    var afterBlock = false;
    text.split(TEX_RE).forEach(function (seg, k) {
      if (!seg) return;
      if (afterBlock && !(k % 2)) { seg = seg.replace(/^\s*[\u061B\u060C.:]\s*/, ''); afterBlock = false; if (!seg) return; }
      if (k % 2) {
        var f = document.createElement('span'); f.className = 'fx'; f.dir = 'ltr';
        var src = seg.slice(1, -1), big = src.length > 26;   /* long formulas get a centred line of their own */
        if (big) { f.className += ' blk'; afterBlock = true; }
        if (window.katex) { try { window.katex.render(src, f, { displayMode: big, throwOnError: false, strict: false }); } catch (e) { f.textContent = src; } }
        else f.textContent = src;
        el.appendChild(f);
      } else inlineText(el, seg);
    });
    return el;
  }
  function inlineText(el, text) {
    text.split(EN_RE).forEach(function (part, n) {
      if (!part) return;
      if (n % 2) { var sp = document.createElement('span'); sp.className = 'en'; sp.dir = 'ltr'; sp.textContent = part; el.appendChild(sp); return; }
      part.split(LAT_RE).forEach(function (r, j) {
        if (!r) return;
        if (j % 2) { var l = document.createElement('span'); l.className = 'lat'; l.dir = 'ltr'; l.textContent = r; el.appendChild(l); return; }
        r.split(NUM_RE).forEach(function (q, m) {
          if (!q) return;
          if (m % 2) { var b = document.createElement('span'); b.className = 'num'; b.textContent = q; el.appendChild(b); }
          else el.appendChild(document.createTextNode(q));
        });
      });
    });
    return el;
  }
  function el(tag, cls) { var e = document.createElement(tag); if (cls) e.className = cls; return e; }
  function renderFull(box, text) {
    var sents = text.split(/(?<=[.!\u061F])\s+/), para = '', n = 0;
    function flush() { if (para) box.appendChild(inline(el('p'), para)); para = ''; n = 0; }
    sents.forEach(function (t) {
      para = para ? para + ' ' + t : t; n++;
      if (n >= 3 || para.length > 240) flush();
    });
    flush();
  }
  function renderSum(box, text) {
    var list = null;
    text.split('\n').forEach(function (line) {
      line = line.trim(); if (!line) return;
      var m = line.match(/^([0-9\u06F0-\u06F9]+)[.)\-]\s*(.*)$/), u = line.match(/^[-\u2022*]\s*(.*)$/), w = line.match(/^!\s*(.*)$/);
      var kind = m ? 'ol' : u ? 'ul' : null;
      if (kind) {
        if (!list || list.tagName.toLowerCase() !== kind) { list = el(kind); box.appendChild(list); }
        list.appendChild(inline(el('li'), m ? m[2] : u[1]));
      } else { list = null; box.appendChild(inline(el('p', w ? 'warn' : ''), w ? w[1] : line)); }
    });
  }
  /* ---------------- editing notes (laptop presenter window only) ----------------
     The editor shows the stored text exactly as it is in content/fa.json: the full notes, or with the summary switch on the
     summary lines ("1. " numbered, "- " bullet, "! " caveat). Saving goes through serve.py, which writes fa.json and rebuilds
     index.html; the way notes are displayed does not change. */
  var edit = null;
  function setNoteText(key, kind, text) {
    var full = $('aside.notes[data-key="' + key + '"]');
    if (!full) return;
    if (kind === 'notes') { full.textContent = text; return; }
    var sum = full.nextElementSibling && full.nextElementSibling.classList.contains('notes-sum') ? full.nextElementSibling : null;
    if (!text.trim()) { if (sum) sum.remove(); return; }
    if (!sum) { sum = document.createElement('aside'); sum.className = 'notes-sum'; full.parentNode.insertBefore(sum, full.nextSibling); }
    sum.textContent = text;
  }
  function edMsg(t, ok) { var m = $('#pv-edmsg'); m.textContent = t || ''; m.className = ok ? 'ok' : ''; }
  function closeEditor() {
    edit = null;
    $('#presenter').classList.remove('editing'); $('#pv-editor').hidden = true;
    renderNotes(slides[state.i]);
  }
  function initEditor() {
    var btn = $('#pv-edit'), ta = $('#pv-editor');
    if (REMOTE) return;
    if (!SERVED) { btn.disabled = true; btn.title = CFG.editNeedsServer; return; }
    btn.addEventListener('click', function () {
      var s = slides[state.i], full = $('.notes', s), key = full && full.getAttribute('data-key');
      if (!key) return;
      var sum = $('.notes-sum', s), kind = sumOn ? 'sum' : 'notes';
      edit = { key: key, kind: kind };
      ta.value = kind === 'sum' ? (sum ? sum.textContent : '') : full.textContent;
      edMsg('');
      $('#presenter').classList.add('editing'); ta.hidden = false; ta.focus();
    });
    $('#pv-cancel').addEventListener('click', closeEditor);
    var save = function () {
      if (!edit) return;
      var e = edit, text = ta.value.replace(/\r\n?/g, '\n').trim();
      fetch(api('notes'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: e.key, kind: e.kind, text: text }) })
        .then(function (r) { return r.json().then(function (j) { if (!r.ok || !j.ok) throw new Error(j.error || r.status); }); })
        .then(function () {
          setNoteText(e.key, e.kind, text);
          if (window.opener && !window.opener.closed) window.opener.postMessage({ type: 'deck-notes', key: e.key, kind: e.kind, text: text }, '*');
          closeEditor(); edMsg('✓', true); setTimeout(function () { edMsg(''); }, 2500);
        })
        .catch(function (err) { edMsg(CFG.editFail + ' (' + (err && err.message || '') + ')'); });
    };
    $('#pv-save').addEventListener('click', save);
    ta.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') { ev.preventDefault(); closeEditor(); }
      else if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); save(); }
    });
  }

  var sumOn = false;
  try { sumOn = localStorage.getItem('deck-notes-sum') === '1'; } catch (e) {}
  function renderNotes(s) {
    if (edit) return;   /* an open editor keeps its text; the notes are drawn again when it closes */
    var box = $('#pv-notes'), full = $('.notes', s), sum = $('.notes-sum', s);
    box.textContent = '';
    var useSum = sumOn && sum && sum.textContent.trim();
    box.classList.toggle('sum', !!useSum);
    if (useSum) renderSum(box, sum.textContent.trim());
    else renderFull(box, full ? full.textContent.trim() : '');
    box.parentNode.scrollTop = 0;
  }
  function setSumButton() { $('#pv-sum').setAttribute('aria-pressed', sumOn ? 'true' : 'false'); }
  function updatePresenter() {
    var s = slides[state.i];
    $('#pv-title').textContent = s.getAttribute('data-title') || '';
    renderNotes(s);
    var isBackup = s.classList.contains('backup');
    $('#pv-count').textContent = isBackup ? CFG.backupLabel + ' ' + fa(state.i - mainCount + 1) + ' ' + CFG.of + ' ' + fa(backupCount) : fa(state.i + 1) + ' ' + CFG.of + ' ' + fa(mainCount);
    var part = isBackup ? 0 : (+s.getAttribute('data-part') || 0);
    $('#pv-part').textContent = part ? CFG.partLabel + ' ' + fa(part) + ' · ' + CFG.parts[part - 1] : '';
    var nx = state.step < steps[state.i] ? { i: state.i, step: state.step + 1 } : { i: Math.min(slides.length - 1, state.i + 1), step: 0 };
    if (frames[0] && frames[0].contentWindow) frames[0].contentWindow.postMessage({ type: 'deck-state', i: state.i, step: state.step }, '*');
    if (frames[1] && frames[1].contentWindow) frames[1].contentWindow.postMessage({ type: 'deck-state', i: nx.i, step: nx.step }, '*');
    var nt = $('#pv-ntitle');
    if (nt) nt.textContent = (nx.i === state.i && nx.step === state.step) ? CFG.endLabel : nx.i === state.i ? CFG.sameSlide : (slides[nx.i].getAttribute('data-title') || '');
    updateVideoPanel();
    if (lastVideo && frames[0] && frames[0].contentWindow) frames[0].contentWindow.postMessage({ type: 'deck-video', video: lastVideo }, '*');
    if (t0 === null && (state.i > 0 || state.step > 0)) t0 = Date.now();
  }
  function initPresenter() {
    var base = location.pathname.split('/').pop() || 'index.html';
    frames = $$('#presenter iframe');
    frames.forEach(function (f) { f.src = base + '?embed=1'; });
    setInterval(function () {
      var s = t0 === null ? 0 : Math.floor((Date.now() - t0) / 1000);
      var mm = String(Math.floor(s / 60)).padStart(2, '0'), ss = String(s % 60).padStart(2, '0');
      $('#pv-clock').textContent = mm + ':' + ss;
    }, 500);
    $('#pv-reset').addEventListener('click', function () { t0 = Date.now(); this.blur(); });
    /* the summary switch changes only when it is clicked; its state survives slide changes and reopening */
    setSumButton();
    $('#pv-sum').addEventListener('click', function () {
      sumOn = !sumOn;
      try { localStorage.setItem('deck-notes-sum', sumOn ? '1' : '0'); } catch (e) {}
      setSumButton(); renderNotes(slides[state.i]); this.blur();
    });
    /* video controls act on the audience window; blur afterwards so Space and the arrows keep driving the slides */
    $$('#pv-video button').forEach(function (b) {
      b.addEventListener('click', function () {
        var a = b.getAttribute('data-v');
        command(a === 'back' ? { cmd: 'video', act: 'nudge', d: -5 } : a === 'fwd' ? { cmd: 'video', act: 'nudge', d: 5 } : { cmd: 'video', act: a });
        b.blur();
      });
    });
    var seek = $('#pv-vseek');
    seek.addEventListener('input', function () { seeking = true; if (lastVideo && lastVideo.d) command({ cmd: 'video', act: 'seek', t: seek.value / 1000 * lastVideo.d }); });
    seek.addEventListener('change', function () { seeking = false; seek.blur(); });
    if (window.opener) window.opener.postMessage({ type: 'deck-hello' }, '*');
    if (REMOTE) initRemote();
    initEditor();
    initPointerPad();
    updatePresenter();
  }

  /* pointer: press and drag on the current-slide picture (finger on the phone, mouse in the laptop presenter window) and a dot
     follows on the audience screen. The picture is letterboxed inside its box, so positions are mapped to the 16:9 slide. */
  function initPointerPad() {
    var pad = $('#presenter .pv.cur'), dot = document.createElement('i'), busy = false, pending = null, down = false;
    dot.id = 'pv-dot'; pad.appendChild(dot);
    /* phone: only the newest position is sent and never two requests at once, so positions arrive in order */
    var flush = function () {
      if (busy || !pending) return;
      var c = pending; pending = null; busy = true;
      var over = false, fin = function () { if (over) return; over = true; clearTimeout(t); busy = false; flush(); };
      var t = setTimeout(fin, 1500);   /* a request that hangs must not block the pointer for good */
      try { fetch(api('cmd'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ c: c }) }).then(fin, fin); } catch (e) { fin(); }
    };
    var send = function (c) { if (REMOTE) { pending = c; flush(); } else command(c); };
    var point = function (e, off) {
      var r = pad.getBoundingClientRect(), k = Math.min(r.width / 1920, r.height / 1080);
      var w = 1920 * k, h = 1080 * k, ox = (r.width - w) / 2, oy = (r.height - h) / 2;
      var x = Math.max(0, Math.min(1, (e.clientX - r.left - ox) / w)), y = Math.max(0, Math.min(1, (e.clientY - r.top - oy) / h));
      if (!off) dot.style.transform = 'translate(' + (ox + x * w) + 'px,' + (oy + y * h) + 'px)';
      dot.classList.toggle('on', !off);
      send(off ? { cmd: 'pointer', off: true, n: Date.now() } : { cmd: 'pointer', x: +x.toFixed(4), y: +y.toFixed(4), n: Date.now() });
    };
    var start = function (e) { down = true; if (REMOTE) keepAwake(); point(e); };
    var end = function (e) { if (down) { down = false; point(e, true); } };
    if (window.PointerEvent) {
      pad.addEventListener('pointerdown', function (e) { if (e.button > 0) return; try { pad.setPointerCapture(e.pointerId); } catch (x) {} start(e); e.preventDefault(); });
      pad.addEventListener('pointermove', function (e) { if (down) { point(e); e.preventDefault(); } });
      ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(function (t) { pad.addEventListener(t, end); });
    } else {   /* older mobile browsers */
      var first = function (e) { return e.touches[0] || e.changedTouches[0]; };
      pad.addEventListener('touchstart', function (e) { start(first(e)); e.preventDefault(); }, { passive: false });
      pad.addEventListener('touchmove', function (e) { if (down) { point(first(e)); e.preventDefault(); } }, { passive: false });
      ['touchend', 'touchcancel'].forEach(function (t) { pad.addEventListener(t, function (e) { end(first(e)); }); });
    }
    pad.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    pad.addEventListener('dragstart', function (e) { e.preventDefault(); });
  }

  /* ---------------- phone presenter (remote) ---------------- */
  var lock = null, noSleep = null;
  function keepAwake() {
    /* Wake Lock needs a secure page; on a plain http LAN address a tiny silent looping video keeps the screen on instead */
    if (navigator.wakeLock && window.isSecureContext) {
      if (!lock) { lock = true; navigator.wakeLock.request('screen').then(function (l) { lock = l; l.addEventListener('release', function () { lock = null; }); }).catch(function () { lock = null; }); }
      return;
    }
    if (!noSleep) {
      noSleep = document.createElement('video');
      noSleep.setAttribute('playsinline', ''); noSleep.setAttribute('muted', ''); noSleep.muted = true; noSleep.loop = true;
      noSleep.src = 'assets/video/depth-demo-title.mp4';
      noSleep.className = 'nosleep';
      document.body.appendChild(noSleep);
    }
    if (noSleep.paused) { var p = noSleep.play(); if (p && p.catch) p.catch(function () {}); }
  }
  /* ---------------- S Pen button as a clicker (phone view, optional) ----------------
     A web page cannot read the S Pen button itself. Air actions turn a press into something the phone does, and we listen for
     every form that can reach a page:
       - media keys (Air actions "Media": single press = play/pause, double press = next track) through the Media Session API,
         which needs a sound to be "playing", so a silent loop runs while pen mode is on;
       - browser Back / Forward (the usual actions for a browser app), caught by keeping the page in the middle of three
         history entries and stepping back to the middle after each move;
       - ordinary keys (Page Down / arrows), already handled by onKey.
     First kind of press -> next, second kind -> previous. A toast names what arrived, so the Air actions can be set to match. */
  function initPen(buzz) {
    var btn = $('#pv-pen'), toast = $('#pv-pentoast'), on = false, armed = false, audio = null, last = 0, tt = null, skipPop = 0;
    try { on = localStorage.getItem('deck-pen') === '1'; } catch (e) {}
    var say = function (t) { toast.textContent = t; toast.classList.add('show'); clearTimeout(tt); tt = setTimeout(function () { toast.classList.remove('show'); }, 1400); };
    var act = function (dir, via) {
      if (!on) return;
      var now = Date.now(); if (now - last < 180) return; last = now;   /* play and pause can both fire for one press */
      buzz(); command({ cmd: dir });
      say((dir === 'next' ? CFG.penNext : CFG.penPrev) + ' · ' + via);
    };
    var silentUrl = function () {   /* 20 s of silence, 8-bit mono WAV */
      var rate = 8000, n = rate * 20, b = new Uint8Array(44 + n), v = new DataView(b.buffer), w = function (o, s) { for (var i = 0; i < s.length; i++) b[o + i] = s.charCodeAt(i); };
      w(0, 'RIFF'); v.setUint32(4, 36 + n, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
      v.setUint32(24, rate, true); v.setUint32(28, rate, true); v.setUint16(32, 1, true); v.setUint16(34, 8, true); w(36, 'data'); v.setUint32(40, n, true);
      b.fill(128, 44);
      return URL.createObjectURL(new Blob([b], { type: 'audio/wav' }));
    };
    var arm = function () {   /* needs a tap: sound and history entries made without one are ignored by the browser */
      if (!on) return;
      if (!audio) { audio = new Audio(silentUrl()); audio.loop = true; }
      var p = audio.play(); if (p && p.catch) p.catch(function () {});
      if ('mediaSession' in navigator) {
        var ms = navigator.mediaSession, keep = function () { ms.playbackState = 'playing'; if (audio.paused) { var q = audio.play(); if (q && q.catch) q.catch(function () {}); } };
        try { ms.metadata = new MediaMetadata({ title: document.title, artist: 'S Pen' }); } catch (e) {}
        var set = function (name, dir) { try { ms.setActionHandler(name, function () { keep(); act(dir, name); }); } catch (e) {} };
        set('play', 'next'); set('pause', 'next'); set('stop', 'next'); set('nexttrack', 'prev'); set('previoustrack', 'prev');
        set('seekforward', 'next'); set('seekbackward', 'prev');
        ms.playbackState = 'playing';
      }
      if (!armed) {
        armed = true;
        try {
          history.replaceState({ pen: 0 }, ''); history.pushState({ pen: 1 }, ''); history.pushState({ pen: 2 }, '');
          skipPop++; history.go(-1);
        } catch (e) {}
      }
    };
    window.addEventListener('popstate', function (e) {
      var s = e.state && e.state.pen;
      if (s === undefined || s === null) return;
      if (skipPop > 0 && s === 1) { skipPop--; return; }   /* our own step back to the middle entry */
      if (s === 0) { skipPop++; history.go(1); act('next', 'back'); }
      else if (s === 2) { skipPop++; history.go(-1); act('prev', 'forward'); }
    });
    var show = function () { btn.setAttribute('aria-pressed', on ? 'true' : 'false'); };
    btn.addEventListener('click', function () {
      on = !on; show();
      try { localStorage.setItem('deck-pen', on ? '1' : '0'); } catch (e) {}
      if (on) arm(); else if (audio) audio.pause();
      toast.textContent = on ? CFG.penOn : CFG.penOff; toast.classList.add('show'); clearTimeout(tt); tt = setTimeout(function () { toast.classList.remove('show'); }, 2600);
      btn.blur();
    });
    show();
    if (on) document.addEventListener('click', arm, { once: true });
    document.addEventListener('visibilitychange', function () { if (on && armed && document.visibilityState === 'visible' && audio && audio.paused) { var p = audio.play(); if (p && p.catch) p.catch(function () {}); } });
  }

  function initRemote() {
    var buzz = function () { try { if (navigator.vibrate) navigator.vibrate(12); } catch (e) {} };
    var tap = function (id, fn) {
      $(id).addEventListener('click', function (e) { e.preventDefault(); buzz(); keepAwake(); fn(); this.blur(); });
    };
    tap('#presenter .pv.next', function () { command({ cmd: 'next' }); });
    tap('#pv-prev', function () { command({ cmd: 'prev' }); });
    tap('#pv-list', function () { toggleOverview(); });
    tap('#pv-fs', function () {
      var d = document.documentElement;
      if (!document.fullscreenElement) { if (d.requestFullscreen) d.requestFullscreen().catch(function () {}); }
      else document.exitFullscreen();
    });
    var size = 17;
    try { size = +localStorage.getItem('deck-remote-font') || 17; } catch (e) {}
    var setSize = function (v) {
      size = Math.max(13, Math.min(28, v));
      document.documentElement.style.setProperty('--nf', size + 'px');
      try { localStorage.setItem('deck-remote-font', size); } catch (e) {}
    };
    setSize(size);
    $$('#presenter [data-fsz]').forEach(function (b) {
      b.addEventListener('click', function () { setSize(size + (+b.getAttribute('data-fsz'))); b.blur(); });
    });
    initPen(buzz);

    var cl = document.createElement('button');
    cl.type = 'button'; cl.className = 'ovclose'; cl.textContent = CFG.closeLabel;
    cl.addEventListener('click', function () { toggleOverview(false); });
    $('#overview').appendChild(cl);
    document.addEventListener('click', keepAwake, { once: true });
    setConn('bad');
  }

  /* ---------------- input ---------------- */
  /* the deck reads right to left, so the left arrow is 'next' and the right arrow is 'previous' */
  function onKey(e) {
    if (e.target && e.target.tagName === 'TEXTAREA') return;   /* typing in the notes editor must not move the slides */
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key;
    if (e.code && /^Key[A-Z]$/.test(e.code)) k = e.code.charAt(3).toLowerCase(); /* letter keys by position, so they also work with the Persian layout active */
    if ($('#overview').classList.contains('open')) { if (k === 'Escape' || k === 'o' || k === 'O') toggleOverview(false); return; }
    if (k === 'ArrowLeft' || k === 'ArrowDown' || k === 'PageDown' || k === ' ' || k === 'Enter' || k === 'n') { e.preventDefault(); command({ cmd: 'next' }); }
    else if (k === 'ArrowRight' || k === 'ArrowUp' || k === 'PageUp' || k === 'Backspace') { e.preventDefault(); command({ cmd: 'prev' }); }
    else if (k === 'Home') command({ cmd: 'goto', i: 0 });
    else if (k === 'End') command({ cmd: 'goto', i: mainCount - 1 });
    else if (k === 'b' || k === 'B') command({ cmd: 'goto', i: mainCount });
    else if (k === 'o' || k === 'O' || k === 'g' || k === 'G') toggleOverview();
    else if ((k === 'f' || k === 'F') && MODE === 'main') { if (!document.fullscreenElement) document.documentElement.requestFullscreen(); else document.exitFullscreen(); }
    else if ((k === 'p' || k === 'P') && MODE === 'main') { presenterWin = window.open((location.pathname.split('/').pop() || 'index.html') + '?presenter=1', 'deck-presenter', 'width=1400,height=860'); }
    else if (k === 'k' || k === 'K') command({ cmd: 'video', act: 'toggle' });
    else if (k === 'j' || k === 'J') command({ cmd: 'video', act: 'nudge', d: -5 });
    else if (k === 'l' || k === 'L') command({ cmd: 'video', act: 'nudge', d: 5 });
    else if (k === 'r' || k === 'R') command({ cmd: 'video', act: 'restart' });
    else if (k === 'm' || k === 'M') command({ cmd: 'video', act: 'mute' });
    else if (k === '?' || k === 'h' || k === 'H') flashHelp(5000);
  }
  function flashHelp(ms) { var h = $('#help'); h.classList.add('show'); setTimeout(function () { h.classList.remove('show'); }, ms); }

  /* ---------------- boot ---------------- */
  document.body.classList.add('no-anim');
  buildChrome(); renderMath(); renderCharts(); buildOverview();
  fit(); window.addEventListener('resize', fit);
  var start = parseInt((location.hash || '').replace('#', ''), 10);
  var startI = isNaN(start) ? 0 : Math.max(0, Math.min(slides.length - 1, start - 1));
  show(startI, qs.has('all') ? steps[startI] : 0, 1, true);
  requestAnimationFrame(function () { requestAnimationFrame(function () { document.body.classList.remove('no-anim'); }); });

  if (MODE === 'main') $$('video:not([data-ambient])').forEach(function (v) {
    ['play', 'pause', 'timeupdate', 'seeking', 'seeked', 'volumechange', 'loadedmetadata', 'ended'].forEach(function (ev) { v.addEventListener(ev, broadcastVideo); });
  });
  else $$('video').forEach(function (v) { v.removeAttribute('controls'); v.muted = true; });

  if (MODE === 'embed') {
    if (window.parent !== window) {
      window.parent.postMessage({ type: 'deck-ready' }, '*');
      document.addEventListener('keydown', function (e) {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        window.parent.postMessage({ type: 'deck-key', key: e.key, code: e.code }, '*');
      });
    }
  } else {
    document.addEventListener('keydown', onKey);
    if (MODE === 'presenter') initPresenter(); else flashHelp(4500);
  }
  connectServer();
  window.__deck = { show: show, state: state, steps: steps, count: slides.length };
})();
