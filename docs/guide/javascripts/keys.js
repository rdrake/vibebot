/* IRC client keys for the guide.
   Sections are the client's networks and pages its windows: Alt+1..9 jumps
   to a section as irssi jumps to a window, and Alt+Up/Down and
   Alt+Shift+Up/Down step through pages and sections as Seance and The Lounge
   step through windows and networks. Material already binds / s f (search)
   and n p (next and previous page); ? and Alt+/ list everything.
   Keys are matched on e.code, since Option on a Mac turns Alt+1 into "¡".
   The script runs once: instant navigation swaps the body but keeps
   document and <html>, so the listener and the theme attribute both survive. */

(() => {
  const root = document.documentElement;
  const SECRET = "keeki";
  let typed = "";

  const KEYS = [
    ["Alt+1 … 9", "jump to section 1–9 (the top tabs)"],
    ["Alt+↑ / Alt+↓", "previous / next page"],
    ["Alt+Shift+↑ / ↓", "previous / next section"],
    ["n  .  /  p  ,", "next / previous page"],
    ["/  s  f", "search"],
    ["Esc", "close search or this help"],
    ["?  Alt+/", "this help"],
  ];

  function typing(e) {
    const el = e.target;
    return el instanceof HTMLElement &&
      (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
  }

  function go(link) {
    if (link) link.click();
  }

  function tabs() {
    return [...document.querySelectorAll(".md-tabs__link")];
  }

  function section(step) {
    const all = tabs();
    const at = all.findIndex((a) => a.closest(".md-tabs__item--active"));
    go(all[Math.min(Math.max(at + step, 0), all.length - 1)]);
  }

  // The footer's own link, so instant navigation sees the click.
  function page(rel) {
    go(document.querySelector(`.md-footer__link--${rel}`));
  }

  let help;
  function toggleHelp() {
    if (!help) {
      help = document.createElement("dialog");
      help.className = "vb-keys";
      help.setAttribute("aria-label", "Keyboard shortcuts");
      help.innerHTML =
        '<p class="vb-keys__head">/help keys</p><dl>' +
        KEYS.map(([k, v]) => `<dt><kbd>${k}</kbd></dt><dd>${v}</dd>`).join("") +
        '</dl><p class="vb-keys__end">End of /HELP. Esc to close.</p>';
      help.addEventListener("click", (e) => {
        if (e.target === help) help.close();
      });
      document.body.append(help);
    }
    if (help.open) help.close();
    else help.showModal();
  }

  function toggleTheme() {
    const on = root.dataset.vbTheme !== SECRET;
    if (on) root.dataset.vbTheme = SECRET;
    else delete root.dataset.vbTheme;
    try {
      if (on) localStorage.setItem("vb-theme", SECRET);
      else localStorage.removeItem("vb-theme");
    } catch (e) {}
  }

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || typing(e)) return;

    if (e.altKey) {
      let handled = true;
      if (/^Digit[1-9]$/.test(e.code)) go(tabs()[Number(e.code.slice(5)) - 1]);
      else if (e.code === "ArrowUp" && e.shiftKey) section(-1);
      else if (e.code === "ArrowDown" && e.shiftKey) section(1);
      else if (e.code === "ArrowUp") page("prev");
      else if (e.code === "ArrowDown") page("next");
      else if (e.code === "Slash") toggleHelp();
      else handled = false;
      if (handled) e.preventDefault();
      return;
    }

    if (e.key === "?") {
      e.preventDefault();
      toggleHelp();
      return;
    }

    if (e.key.length === 1) {
      typed = (typed + e.key.toLowerCase()).slice(-SECRET.length);
      if (typed === SECRET) {
        typed = "";
        toggleTheme();
      }
    }
  });
})();
