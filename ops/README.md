# Swapping production to the megakernel (and back)

What we did on 2026-10-20, on a single 4× B200 that serves users. A second machine could test without any downtime; we accepted a short night-time restart instead.

1. **Launcher override.** The production launcher sources an env file and, if it sets `SERVE_OVERRIDE`, `exec`s that script. Boot, watchdog restarts and manual restarts therefore all pick the megakernel launcher while the file exists. Without the file, they go back to the standard one.
2. **Close the tunnel.** Users stop arriving.
   - Kill the tunnel's supervision loop first, then the connector.
   - **Wait for the connector to exit** (it has a grace period of several seconds). Checking after 3 s once made us abort a window and, worse, think the tunnel was still up while it was dying.
3. **Restart with the override.** Wait for `/health` (18 minutes on the first megakernel boot).
4. **Gates against the production port:** megakernel active on all ranks, KL ≤ 0.03, GSM8K ≥ 0.94 on 500, functional and leak checks ([`tools/gates.sh`](../tools/gates.sh)). Any failure removes the override, restarts the standard path, and checks it the same way.
5. **Measure before opening.** With users inside, the numbers are not clean.
6. **Reopen.**
   - Start the tunnel loop.
   - Count it as open only when an unauthenticated request from outside reaches the server (401). A live connector process alone does not count.

Rollback at any time: move the env file away and restart (~5 min on warm boots of the standard path).

Our timeline: tunnel closed 01:20:26 UTC → ready 01:38:24 → gates passed 01:39:45 → measured → open 01:40:11. That is ~20 minutes down, most of it the first megakernel boot.
