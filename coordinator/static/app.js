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

  function refresh() {
    if (inflight) { return Promise.resolve(); }
    inflight = true;
    return fetch("/fragments/fleet", { cache: "no-store" }).then(function (res) {
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

  if (panel()) { showJob(); }
  setInterval(function () { if (!document.hidden) { refresh(); } }, REFRESH_MS);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) { refresh(); } });
})();
