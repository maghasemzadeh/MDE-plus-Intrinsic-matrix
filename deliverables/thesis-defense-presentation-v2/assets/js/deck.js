/* Thesis defense deck engine — no dependencies except KaTeX (bundled, optional).
   Modes:  (default) audience view   ?presenter=1 presenter view   ?embed=1 preview frame */
(function () {
  'use strict';

  var qs = new URLSearchParams(location.search);
  var MODE = qs.has('presenter') ? 'presenter' : (qs.has('embed') ? 'embed' : 'main');
  document.body.classList.add(MODE);

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

  function show(i, step, dir, silent) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    step = Math.max(0, Math.min(steps[i], step));
    var prev = slides[state.i], next = slides[i];
    if (prev !== next) {
      stage.style.setProperty('--dir', dir >= 0 ? 1 : -1);
      prev.classList.remove('active'); prev.classList.add('leaving');
      setTimeout(function () { prev.classList.remove('leaving'); }, 500);
      $$('video', prev).forEach(function (v) { v.pause(); });
      next.classList.add('active');
      if (MODE === 'main') $$('video[data-autoplay]', next).forEach(function (v) { try { v.currentTime = 0; var p = v.play(); if (p && p.catch) p.catch(function () {}); } catch (e) {} });
    } else next.classList.add('active');
    state.i = i; state.step = step;
    applyFrags(next, step);
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
  }

  /* ---------------- sync between windows ---------------- */
  var presenterWin = null;
  function broadcast() {
    var msg = { type: 'deck-state', i: state.i, step: state.step };
    if (MODE === 'main' && presenterWin && !presenterWin.closed) presenterWin.postMessage(msg, '*');
    if (MODE === 'presenter') updatePresenter();
  }
  function command(c) {
    if (MODE === 'presenter' && window.opener && !window.opener.closed) { window.opener.postMessage({ type: 'deck-cmd', c: c }, '*'); return; }
    if (c.cmd === 'next') next(); else if (c.cmd === 'prev') prev(); else if (c.cmd === 'goto') show(c.i, c.step || 0, c.i >= state.i ? 1 : -1);
  }
  window.addEventListener('message', function (ev) {
    var d = ev.data || {};
    if (d.type === 'deck-state') { show(d.i, d.step, d.i >= state.i ? 1 : -1, MODE !== 'presenter'); if (MODE === 'presenter') updatePresenter(); }
    else if (d.type === 'deck-cmd' && MODE === 'main') command(d.c);
    else if (d.type === 'deck-hello' && MODE === 'main') { presenterWin = ev.source; broadcast(); }
    else if (d.type === 'deck-ready' && MODE === 'presenter') updatePresenter();
  });

  /* ---------------- presenter view ---------------- */
  var t0 = null, frames = [];
  function updatePresenter() {
    var s = slides[state.i];
    var notes = $('.notes', s);
    $('#pv-title').textContent = s.getAttribute('data-title') || '';
    $('#pv-notes').textContent = notes ? notes.textContent.trim() : '';
    var isBackup = s.classList.contains('backup');
    $('#pv-count').textContent = isBackup ? CFG.backupLabel + ' ' + fa(state.i - mainCount + 1) + ' ' + CFG.of + ' ' + fa(backupCount) : fa(state.i + 1) + ' ' + CFG.of + ' ' + fa(mainCount);
    var nx = state.step < steps[state.i] ? { i: state.i, step: state.step + 1 } : { i: Math.min(slides.length - 1, state.i + 1), step: 0 };
    if (frames[0] && frames[0].contentWindow) frames[0].contentWindow.postMessage({ type: 'deck-state', i: state.i, step: state.step }, '*');
    if (frames[1] && frames[1].contentWindow) frames[1].contentWindow.postMessage({ type: 'deck-state', i: nx.i, step: nx.step }, '*');
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
    $('#pv-reset').addEventListener('click', function () { t0 = Date.now(); });
    if (window.opener) window.opener.postMessage({ type: 'deck-hello' }, '*');
    updatePresenter();
  }

  /* ---------------- input ---------------- */
  function onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key;
    if ($('#overview').classList.contains('open')) { if (k === 'Escape' || k === 'o' || k === 'O') toggleOverview(false); return; }
    if (k === 'ArrowRight' || k === 'ArrowDown' || k === 'PageDown' || k === ' ' || k === 'Enter' || k === 'n') { e.preventDefault(); command({ cmd: 'next' }); }
    else if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'PageUp' || k === 'Backspace') { e.preventDefault(); command({ cmd: 'prev' }); }
    else if (k === 'Home') command({ cmd: 'goto', i: 0 });
    else if (k === 'End') command({ cmd: 'goto', i: mainCount - 1 });
    else if (k === 'b' || k === 'B') command({ cmd: 'goto', i: mainCount });
    else if (k === 'o' || k === 'O' || k === 'g' || k === 'G') toggleOverview();
    else if ((k === 'f' || k === 'F') && MODE === 'main') { if (!document.fullscreenElement) document.documentElement.requestFullscreen(); else document.exitFullscreen(); }
    else if ((k === 'p' || k === 'P') && MODE === 'main') { presenterWin = window.open((location.pathname.split('/').pop() || 'index.html') + '?presenter=1', 'deck-presenter', 'width=1400,height=860'); }
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

  if (MODE === 'embed') {
    if (window.parent !== window) window.parent.postMessage({ type: 'deck-ready' }, '*');
  } else {
    document.addEventListener('keydown', onKey);
    if (MODE === 'presenter') initPresenter(); else flashHelp(4500);
  }
  window.__deck = { show: show, state: state, steps: steps, count: slides.length };
})();
