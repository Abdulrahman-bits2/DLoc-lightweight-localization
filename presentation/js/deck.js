/* =====================================================================
   Minimal, dependency-free slide engine for the DLoc defense deck.
   - fixed 1280x720 stage, scaled to fit the viewport
   - keyboard + click navigation, progress bar, slide counter
   - speaker-notes panel (press S), fullscreen (F), overview hash deep-links
   ===================================================================== */
(function () {
  "use strict";
  var stage = document.getElementById("stage");
  var slides = Array.prototype.slice.call(document.querySelectorAll(".slide"));
  var bar = document.getElementById("bar");
  var counter = document.getElementById("counter");
  var notesBody = document.querySelector("#notes .nbody");
  var i = 0;

  function scale() {
    var s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    stage.style.transform = "scale(" + s + ")";
  }

  function render() {
    slides.forEach(function (sl, n) { sl.classList.toggle("active", n === i); });
    bar.style.width = ((i + 1) / slides.length * 100) + "%";
    counter.textContent = (i + 1) + " / " + slides.length;
    var note = slides[i].querySelector(".notes");
    notesBody.textContent = note ? note.textContent.trim() : "(no notes for this slide)";
    if (location.hash !== "#" + (i + 1)) {
      history.replaceState(null, "", "#" + (i + 1));
    }
  }

  function go(n) {
    i = Math.max(0, Math.min(slides.length - 1, n));
    render();
  }
  function next() { go(i + 1); }
  function prev() { go(i - 1); }

  var typed = "";              // buffer for "type a number + Enter to jump"
  document.addEventListener("keydown", function (e) {
    // numeric jump: collect digits, Enter to go (great for backup slides in Q&A)
    if (e.key >= "0" && e.key <= "9") { typed += e.key; counter.textContent = "→ " + typed; return; }
    if (e.key === "Enter") {
      if (typed) { go(parseInt(typed, 10) - 1); typed = ""; } return;
    }
    if (e.key === "Escape") { typed = ""; render(); return; }
    switch (e.key) {
      case "ArrowRight": case "PageDown": case " ": case "Spacebar":
        e.preventDefault(); next(); break;
      case "ArrowLeft": case "PageUp":
        e.preventDefault(); prev(); break;
      case "Home": e.preventDefault(); go(0); break;
      case "End": e.preventDefault(); go(slides.length - 1); break;
      case "s": case "S":
        document.body.classList.toggle("show-notes"); break;
      case "f": case "F":
        if (!document.fullscreenElement) {
          (document.documentElement.requestFullscreen || function () {}).call(document.documentElement);
        } else { document.exitFullscreen && document.exitFullscreen(); }
        break;
    }
  });

  // click navigation: left third = back, rest = forward (ignore links/buttons)
  stage.addEventListener("click", function (e) {
    if (e.target.closest("a,button")) return;
    var r = stage.getBoundingClientRect();
    if ((e.clientX - r.left) / r.width < 0.25) prev(); else next();
  });

  window.addEventListener("resize", scale);
  window.addEventListener("hashchange", function () {
    var n = parseInt(location.hash.slice(1), 10);
    if (!isNaN(n) && n - 1 !== i) go(n - 1);
  });

  // init
  scale();
  var start = parseInt(location.hash.slice(1), 10);
  if (!isNaN(start)) i = Math.max(0, Math.min(slides.length - 1, start - 1));
  render();
})();
