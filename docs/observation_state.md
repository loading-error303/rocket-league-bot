# Observation/State Documentation (RLGym DefaultObs)

This document describes the exact observation vector produced by the current environment setup in [src/env.py](src/env.py). The environment uses `DefaultObs` with **no zero padding** (`zero_padding=None`), so the observation length depends on the number of cars present.

## Summary

- **Observation builder**: `rlgym.rocket_league.obs_builders.DefaultObs`
- **Normalization**:
  - Position: scaled by `pos_coef` (see below)
  - Orientation angles: scaled by `ang_coef = 1/π`
  - Linear velocity: scaled by `lin_vel_coef`
  - Angular velocity: scaled by `ang_vel_coef`
  - Boost amount: scaled by `boost_coef = 1/100`
  - Boost pad timers: scaled by `pad_timer_coef = 1/10`
- **Ordering**: global ball + pad info → partial self state → self car physics → allies → enemies
- **Inversion for orange team**: If the observing car is orange, the builder uses inverted ball, inverted boost pads, and inverted car physics so both teams see a consistent “blue-side” frame.

## Common Concepts (Defined Once)

### Position (3)
3D position vector `(x, y, z)` in Rocket League world coordinates. In this project, the vector is **normalized** by:

- `pos_coef = [1/SIDE_WALL_X, 1/BACK_NET_Y, 1/CEILING_Z]`

### Forward / Up (3 each)
Unit direction vectors derived from car orientation:

- `forward`: the direction the car is facing
- `up`: the direction perpendicular to the roof (world-space up from the car)

### Linear Velocity (3)
Car or ball velocity `(vx, vy, vz)` in world units per second, **normalized** by:

- `lin_vel_coef = 1/CAR_MAX_SPEED`

### Angular Velocity (3)
Angular velocity vector `(wx, wy, wz)` in radians per second, **normalized** by:

- `ang_vel_coef = 1/CAR_MAX_ANG_VEL`

### Boolean Flags (1 each)
Binary features are encoded as `0` or `1`.

## Observation Vector Structure

The final observation is a **1D numpy array** built by concatenating the following blocks in order.

### 1) Global Ball + Boost Pads
1. **Ball position (3)** → normalized position
2. **Ball linear velocity (3)** → normalized linear velocity
3. **Ball angular velocity (3)** → normalized angular velocity
4. **Boost pad timers (34)** → timers for each pad, normalized by `pad_timer_coef`
5. **Partial self-state (9)** (not physics, just state flags):
   - `is_holding_jump`
   - `handbrake`
   - `has_jumped`
   - `is_jumping`
   - `has_flipped`
   - `is_flipping`
   - `has_double_jumped`
   - `can_flip`
   - `air_time_since_jump`

### 2) Self Car (Physics + Status) — 20 values
From `_generate_car_obs` for the observing car:

1. **Position (3)** → normalized
2. **Forward (3)**
3. **Up (3)**
4. **Linear velocity (3)** → normalized
5. **Angular velocity (3)** → normalized
6. **Status (5)**:
   - `boost_amount` (normalized by `boost_coef`)
   - `demo_respawn_timer`
   - `on_ground` (0/1)
   - `is_boosting` (0/1)
   - `is_supersonic` (0/1)

### 3) Allies (each 20 values)
For each teammate **excluding self**, the same 20-value car block as in section 2 is appended, in arbitrary iteration order from the state.

### 4) Enemies (each 20 values)
For each opponent, the same 20-value car block is appended after all allies.

## Observation Length (No Zero Padding)

`DefaultObs` returns:

- **Total length** = `52 + 20 * num_cars`
  - 52 = 3 (ball pos) + 3 (ball lin vel) + 3 (ball ang vel) + 34 (pads) + 9 (partial self-state)
  - `num_cars` = all cars in the match (self + allies + enemies)

### Common setups
- **1v1** (`num_cars = 2`) → length **92**
- **2v2** (`num_cars = 4`) → length **132**
- **3v3** (`num_cars = 6`) → length **172**

## Where This Is Configured

- Observation builder creation: [src/env.py](src/env.py)
- The normalization coefficients come directly from `common_values` and the `DefaultObs` constructor in [src/env.py](src/env.py).
