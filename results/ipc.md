# IPC intersection-free guarantee vs a penalty method (measured) — World-3 opened

A spring body thrown into a fixed V-wedge at increasing speed (implicit Euler, dt=1/30). **IPC** = the faithful C² log-barrier `b(d) = −(d−d̂)² ln(d/d̂)` + a **CCD-filtered line search** (the step capped so no vertex ever crosses a wall), minimized by projected Newton (`bench/ipc.py`, conformance-gated: barrier shape + b′,b″ vs FD, and the guarantee below). **Penalty** = a classical quadratic penalty `½κ·max(0,−d)²` (finite, no CCD) — the prior soft-constraint approach. Metric: minimum wall distance over the whole trajectory (`< 0` = interpenetration). Run: `python -m bench.run_ipc`.

| impact speed | IPC min wall-distance | penalty min wall-distance |
|---:|---:|---:|
| 4 | +1.627e-02 | -9.064e-02 ⚠ penetrates |
| 8 | +1.490e-02 | -1.284e-01 ⚠ penetrates |
| 14 | +1.413e-02 | -1.789e-01 ⚠ penetrates |
| 20 | +1.369e-02 | -2.012e-01 ⚠ penetrates |
| 30 | +1.273e-02 | -2.750e-01 ⚠ penetrates |

## Observed — `ipc → prior-rigid-engines` (robustness) adjudicated

- **The intersection-free guarantee reproduces:** IPC keeps every wall distance **strictly positive at every impact speed** (1.6e-02, 1.5e-02, 1.4e-02, 1.4e-02, 1.3e-02) — the barrier is `+∞` at contact and the CCD line search caps each step short of a crossing, so penetration is impossible *by construction*, independent of speed. The classical penalty method **tunnels** on 5/5 of the same impacts (min distance goes negative), and worse as the impact hardens — exactly the failure IPC was designed to remove. This is a **guarantee**, not a tuned win: no κ or timestep makes IPC penetrate, whereas the penalty's safety is speed/stiffness-dependent.

_Scope (honest): a MINIMAL faithful 2D IPC — vertex-vs-half-plane contact (linear CCD), a mass-spring body, a fixed wedge. The barrier, the CCD step cap, and the guarantee are the real IPC mechanism; vertex-vs-edge / mesh-mesh CCD, friction, and 3D/GPU scale are the natural extensions and are NOT implemented — so IPC's *speed/throughput* edges over GIPC, ABD, medial-IPC etc. (all GPU-scale claims) stay `unmeasured`. What is adjudicated is IPC's defining ROBUSTNESS claim — the guaranteed intersection-free trajectory — which this harness demonstrates against the penalty baseline it supplants._
