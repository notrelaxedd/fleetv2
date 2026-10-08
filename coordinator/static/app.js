/* fleet-v2 dashboard: refreshes the Fleet screen every 5 seconds and runs the buttons.
   The pages work without this file; it only keeps them fresh and sends the button clicks. */
(function () {
  "use strict";
  var REFRESH_MS = 5000;
  var inflight = false;

  function $(sel, root) { return (root || document).querySelector(sel); }

  // ---- talking to the coordinator
  function send(url, body) {
    var opts = { method: "POST", headers: { "Content-Type": "application/json" } };
    if (body) { opts.body = JSON.stringify(body); }
    return fetch(url, opts).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        return { ok: res.ok, message: res.ok ? data.message : (data.detail || "Something went wrong (" + res.status + ")") };
      });
    }, function () {
      return { ok: false, message: "Cannot reach the coordinator" };
    });
  }

  var toastTimer = null;
  function toast(text, ok) {
    var el = $("#toast");
    if (!el) { return; }
    el.textContent = text;
    el.dataset.tone = ok ? "ok" : "error";
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.hidden = true; }, 6000);
  }

  // ---- refreshing regions
  // Regions are swapped only when their HTML changed. The Assign panel is not a region, so the
  // selects the owner is using are never re-rendered; only their option lists are updated.
  function syncOptions(live, fresh) {
    var wanted = Array.prototype.map.call(fresh.options, function (o) { return [o.value, o.textContent, o.disabled]; });
    var have = Array.prototype.map.call(live.options, function (o) { return [o.value, o.textContent, o.disabled]; });
    if (JSON.stringify(wanted) === JSON.stringify(have) || document.activeElement === live) { return; }
    var keep = live.value;
    live.innerHTML = fresh.innerHTML;
    var stillThere = Array.prototype.some.call(live.options, function (o) { return o.value === keep && !o.disabled; });
    live.value = stillThere ? keep : (live.options[0] ? live.options[0].value : "");
  }

  function apply(html) {
    var doc = new DOMParser().parseFromString(html, "text/html");
    Array.prototype.forEach.call(doc.querySelectorAll("[data-region]"), function (fresh) {
      var name = fresh.getAttribute("data-region");
      if (fresh.tagName === "SELECT") {
        var live = $('select[data-region-select="' + name + '"]');
        if (live) { syncOptions(live, fresh); }
        return;
      }
      var target = $('[data-region="' + name + '"]');
      if (target && target.innerHTML !== fresh.innerHTML) { target.innerHTML = fresh.innerHTML; }
    });
  }

  // The Fleet screen refreshes /fragments/fleet; the Models screen refreshes /fragments/models for the
  // model that is selected on the page (so the selection stays put even when the ranking moves).
  function fragmentUrl() {
    var screen = $("[data-models-screen]");
    if (!screen) { return "/fragments/fleet"; }
    var id = screen.getAttribute("data-selected-id");
    var query = [];
    if (screen.getAttribute("data-market")) { query.push("market=" + encodeURIComponent(screen.getAttribute("data-market"))); }
    if (id) { query.push("id=" + encodeURIComponent(id)); }
    return "/fragments/models" + (query.length ? "?" + query.join("&") : "");
  }

  function refresh() {
    if (inflight) { return Promise.resolve(); }
    inflight = true;
    return fetch(fragmentUrl(), { cache: "no-store" }).then(function (res) {
      if (!res.ok) { throw new Error("status " + res.status); }
      return res.text();
    }).then(function (html) {
      apply(html);
      $("#stale").hidden = true;
    }).catch(function () {
      $("#stale").hidden = false;
    }).then(function () { inflight = false; });
  }

  // ---- the Assign a job panel
  function panel() { return $('[data-panel="assign"]'); }
  function chosenJob() {
    var select = $('[data-field="kind"]');
    return select ? select.options[select.selectedIndex] : null;
  }

  function showJob() {
    var option = chosenJob();
    if (!option) { return; }
    var needsModel = option.dataset.needsModel === "true";
    $("[data-help]", panel()).textContent = option.dataset.hint || "";
    $('[data-field-wrap="model"]', panel()).hidden = !needsModel;
    $('[data-action="assign"]', panel()).disabled = option.disabled;
  }

  function say(text, ok) {
    var line = $("[data-confirm]", panel());
    line.textContent = text;
    line.dataset.tone = ok ? "ok" : "error";
  }

  function assign(button) {
    var option = chosenJob();
    var body = { kind: option.value, target: $('[data-field="worker"]').value };
    if (option.dataset.needsModel === "true") { body.model_id = $('[data-field="model"]').value; }
    button.disabled = true;
    send("/api/jobs", body).then(function (r) {
      say(r.message, r.ok);
      button.disabled = option.disabled;
      if (r.ok) { refresh(); }
    });
  }

  // ---- Models screen: search, backtest and paper-trading buttons
  function reply(which, text, ok) {
    var line = $('[data-reply="' + which + '"]');
    if (!line) { return; }
    line.textContent = text;
    line.dataset.tone = ok ? "ok" : "error";
  }

  function plainMessage(r, fallback) {
    return typeof r.message === "string" && r.message ? r.message : (r.ok ? fallback : "Something went wrong");
  }

  function modelsAction(button, action) {
    var id = button.dataset.modelId;
    var url, body, which = "model", done = "Done";
    if (action === "search-start" || action === "search-stop") {
      url = "/api/search/" + (action === "search-start" ? "start" : "stop");
      if (action === "search-start" && button.dataset.market) { body = { markets: [button.dataset.market] }; }
      which = "search";
    } else if (action === "futures-prices") {
      url = "/api/jobs";
      body = { kind: "futures_prices", target: "auto" };
      which = "search";
    } else if (action === "final-check") {
      url = "/api/models/" + encodeURIComponent(id) + "/final-check";
    } else if (action === "run-backtest") {
      url = "/api/jobs";
      body = { kind: "backtest", model_id: id, target: "auto" };
    } else {
      url = "/api/models/" + encodeURIComponent(id) + "/paper/" + (action === "paper-start" ? "start" : "stop");
    }
    button.disabled = true;
    send(url, body).then(function (r) {
      reply(which, plainMessage(r, done), r.ok);
      button.disabled = false;
      return refresh();
    });
  }

  // ---- buttons (delegated, so they keep working after a region is swapped)
  document.addEventListener("click", function (e) {
    var button = e.target.closest("button[data-action]");
    if (!button || button.disabled) { return; }
    var action = button.dataset.action;
    if (action === "assign") {
      assign(button);
    } else if (action === "pause" || action === "resume") {
      button.disabled = true;
      send("/api/trading/" + action).then(function (r) {
        if (!r.ok) { toast(r.message, false); }
        return refresh();
      });
    } else if (/^(search-(start|stop)|run-backtest|paper-(start|stop)|futures-prices|final-check)$/.test(action)) {
      modelsAction(button, action);
    } else if (action === "run-again") {
      button.disabled = true;
      send("/api/jobs/" + encodeURIComponent(button.dataset.jobId) + "/run-again").then(function (r) {
        toast(r.message, r.ok);
        if (r.ok) { refresh(); } else { button.disabled = false; }
      });
    }
  });

  document.addEventListener("change", function (e) {
    if (e.target.matches('[data-field="kind"]')) { showJob(); }
  });

  // ===== Live trading panel =====
  // The Trading mode pill opens a dialog; /api/live is fetched each time so it is current.
  // Opening is delegated (the pill is swapped every 5 s); the dialog itself is never swapped.
  function livePanel() { return $("#live-panel"); }
  function liveEl(name) { return $('[data-live="' + name + '"]', livePanel()); }

  function liveSay(text, ok) {
    var line = liveEl("reply");
    line.textContent = text || "";
    line.dataset.tone = ok ? "ok" : "error";
  }

  function liveRender(s) {
    var live = s.mode === "live";
    var mode = liveEl("mode");
    mode.textContent = live ? "Live trading: real money" : "Paper trading: no real money";
    mode.dataset.state = live ? "live" : "paper";
    var checks = { env: s.env_allows_live, keys: s.live_keys_present, confirmed: s.confirmed };
    Object.keys(checks).forEach(function (k) {
      var li = $('[data-live-check="' + k + '"]', livePanel());
      li.dataset.ok = checks[k] ? "true" : "false";
      li.setAttribute("aria-label", li.textContent.trim() + ": " + (checks[k] ? "yes" : "no"));
    });
    liveEl("confirm-form").hidden = !!s.confirmed;
    liveEl("withdraw-form").hidden = !s.confirmed;
    liveEl("phrase").placeholder = "Type " + (s.phrase || "TRADE REAL MONEY");
  }

  function liveLoad() {
    return fetch("/api/live", { cache: "no-store" }).then(function (res) {
      if (!res.ok) { throw new Error("status " + res.status); }
      return res.json();
    }).then(liveRender, function () {
      liveEl("mode").textContent = "Cannot read the trading mode";
      liveSay("Cannot reach the coordinator", false);
    });
  }

  function liveOpen() {
    var dlg = livePanel();
    if (!dlg || dlg.open) { return; }
    liveSay("", true);
    liveEl("phrase").value = "";
    if (typeof dlg.showModal === "function") { dlg.showModal(); } else { dlg.setAttribute("open", ""); }
    liveLoad();
  }

  function liveClose() {
    var dlg = livePanel();
    if (dlg && dlg.open) { if (dlg.close) { dlg.close(); } else { dlg.removeAttribute("open"); } }
  }

  function liveSend(button, url, body) {
    button.disabled = true;
    return send(url, body).then(function (r) {
      return liveLoad().then(function () { liveSay(r.message, r.ok); button.disabled = false; });
    });
  }

  document.addEventListener("click", function (e) {
    var button = e.target.closest("button[data-action]");
    if (!button || button.disabled) { return; }
    var action = button.dataset.action;
    if (action === "open-live") { liveOpen(); }
    else if (action === "live-close") { liveClose(); }
    else if (action === "live-withdraw") { liveSend(button, "/api/live/withdraw"); }
  });

  document.addEventListener("submit", function (e) {
    if (!e.target.matches('[data-live="confirm-form"]')) { return; }
    e.preventDefault();
    liveSend($('[data-action="live-confirm"]', e.target), "/api/live/confirm", { confirm: liveEl("phrase").value.trim() });
  });

  document.addEventListener("close", function (e) {
    if (e.target.id !== "live-panel") { return; }
    var pill = $('[data-action="open-live"]');
    if (pill) { pill.focus(); }
  }, true);

  // A click on the dark backdrop (the dialog element itself) closes it too.
  document.addEventListener("click", function (e) {
    if (e.target.id === "live-panel") { liveClose(); }
  });

  if (panel()) { showJob(); }
  setInterval(function () { if (!document.hidden) { refresh(); } }, REFRESH_MS);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) { refresh(); } });
})();
