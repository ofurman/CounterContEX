# MOONS Greedy Trajectory Figures

`moons_trajectories.png` shows the LR discriminator boundary, MOONS train scatter, and the per-step greedy counterfactual paths for near-boundary test points. Each arrow is axis-aligned because one feature is committed per step; blue arrows move feature 0 and red arrows move feature 1. Green markers reached the target class, red markers stalled before the flip.

`moons_blocked_slice.png` zooms into a representative stalled point (test row 128). The panels hold one coordinate fixed at the final state and vary the other coordinate, overlaying the LR target probability with the TabPFN class-conditional density. When the density mode remains outside the shaded flip side, a single-feature MAP commit lands on the wrong side of the boundary and the greedy path stalls.

Generated from 30 plotted near-boundary trajectories; 77 rows were evaluated including the fallback scan, with 76 flipped and 1 stalled.
