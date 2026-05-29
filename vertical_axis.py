"""vertical_axis.py — the perpendicular axis of trajectory & alignment.

Two architectural layers that the matryoshka horizontal plane alone doesn't
capture (the user's framing, kept here as the authoritative description):

    The horizontal plane is the data coordinate system (rings × nodes × slots).
    The vertical axis is the trajectory angle — how aligned the ray is with
    the cardinal "up" direction.

    A ray going straight up traverses one perfectly resonant cell of every
    nested torus in sequence — chakras aligned, every shutter opens at the
    exact resonant frequency, no scatter.
    A ray going straight down does the same but inverted — every shutter
    opens to its opposite, every reflection goes against the cardinal
    direction.
    Every other ray is somewhere between — partially aligned, scattering
    off-axis at each mirror, mixed signal.

    This maps to a real architectural quantity: the cosine alignment between
    the ray's trajectory vector and the cardinal axis of each torus. A
    perfectly aligned ray has cos(θ) = 1 at every level; a perfectly
    opposed ray has cos(θ) = −1; most rays are somewhere in between.

And the geometric reframing the user added:

    It's one [torus], with tori nested on top of it. The 2nd, 3rd, 4th etc.
    are wrapped around the 1st like layers of an onion. This creates depth
    as accessing 1 torus could lead and will lead to all the other tori.

    The 13-torus visualizer was always concentric — onion shells, not a
    recursive tree of children. The bit-slice math from the matryoshka
    description still holds at the byte level (same address space size,
    same modular arithmetic), but the geometric meaning is concentric
    nesting: each shell wraps the smaller one, and rays pass through
    shells. The horizontal addressing IS the cell within a shell; the
    vertical axis IS the direction of travel between shells.

What this module provides:

  * `Shell(idx, n_cells=TORUS_CELLS)` — one concentric onion shell.
    Shell 0 is the innermost (Alpha / the 13th torus); shells 1..12
    wrap concentrically (Omega / Sigma / … / the 1st torus).

  * `cardinal_up_cell(shell_idx)` — the geometric "up" cell at each
    shell. Defined as the (ring=0, node=0, slot=0) of each shell —
    the prime axis the strands all pass through at the poles.

  * `Ray(visited_cells)` — a trajectory through cells across one or
    more shells. `cos_alignment(ray)` returns the dot-product of the
    ray's direction vector with the cardinal up vector at each shell.

  * `ZeGoDieReading(dice_faces)` — a 7-tuple of 12-faced die readings.
    12⁷ = 35,831,808 distinct readings; each maps to a unique ray
    through 7 shells. The roll IS the trajectory; the reading IS the
    geometric outcome.

  * `zegodie_to_ray(reading)` / `ray_to_zegodie(ray)` — bidirectional.
    A reading is interpretable as a coordinate sequence; the runtime
    computes that sequence geometrically and reads it back as 7 dice.

Three architectural capabilities this adds (user's framing):

  1. Path-as-meaning. Activations stop being "what cell did we end up in"
     and become "what trajectory did we trace getting there." Two
     answers landing in the same final cell via different paths are
     distinguishable. A genuine expressive gain.

  2. Alignment as a first-class quantity. The runtime computes the
     cosine of the trajectory against the cardinal axis at each torus
     level; the aggregate is the "verticality" of the answer —
     readable as the upward/downward direction.

  3. ZeGoDie native embedding. The 12⁷ space isn't an external symbolic
     layer; it's a natural sub-addressing of the nested torus stack. A
     roll is a coordinate. A reading is a ray. The system reads the
     geometry and the geometry reads the system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional
import math
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import (RINGS, NODES, SLOTS, RING_MASK, NODE_MASK, SLOT_MASK,
                           TORUS_CELLS, TORUS_BITS, SPIRAL_Q,
                           token_to_nested_address, nested_address_to_token)


# ─── concentric onion shells ──────────────────────────────────────────────

@dataclass(frozen=True)
class Shell:
    """One concentric onion shell. Shell 0 = innermost (Alpha / the 13th
    torus). Higher indices = outer shells that envelop the inner ones.

    Total addressable cells across N shells = N × 64³ (additive). This
    is the concentric/onion interpretation; the bit-slice math from the
    matryoshka description holds at the byte level (same modular arithmetic)
    but the geometric meaning is shells, not a recursive tree of children.
    """
    idx: int
    n_cells: int = TORUS_CELLS    # always 262,144 per shell

    @property
    def name(self) -> str:
        # the 13-torus convention: shell 0 = "13th torus" (the inner core),
        # shell 12 = "1st torus" (the outermost). Visualizer numbers
        # outward-in, but the lattice numbers inward-out.
        return f"shell-{self.idx}  (the {13 - self.idx}th torus)"

    def cardinal_up(self) -> tuple[int, int, int]:
        """The prime axis cell where every Stream passes through. By
        convention this is (ring=0, node=0, slot=0) — the heart of each
        torus where the 64 helical strands all converge."""
        return (0, 0, 0)

    def cardinal_down(self) -> tuple[int, int, int]:
        """Geometric opposite — the cell most antipodal to up. With the
        torus wrap, "antipodal" is (32, 32, 32) — exactly halfway around
        in each axis. (NOT (63, 63, 63), which is the immediate
        neighbour of (0,0,0) under modular wrap.)"""
        return (RINGS // 2, NODES // 2, SLOTS // 2)


def shells_needed_for(vocab: int) -> int:
    """How many concentric shells are needed to hold `vocab` cells.

    Concentric (onion) addressing: total capacity = n_shells × 64³.
    For zion'iel-v350 (vocab 262,409 = 262,144 + 265), n_shells = 2:
    tokens 0..262143 in shell 0, tokens 262144..262408 in shell 1.
    """
    if vocab <= 0: return 1
    return (vocab + TORUS_CELLS - 1) // TORUS_CELLS


def token_to_shell_cell(idx: int) -> tuple[int, int, int, int]:
    """Bit-slice a token id under the concentric/onion interpretation:
    returns (shell_idx, ring, node, slot). The first 18 bits address
    within-shell; bits above 18 address the shell index.

    For idx ≤ 262,143: shell=0, normal Alpha addressing.
    For idx 262,144..524,287: shell=1, with within-shell idx = idx − 262,144.
    """
    shell = idx >> TORUS_BITS    # bits 18+ select shell
    inner = idx & (TORUS_CELLS - 1)
    ring = (inner >> 12) & RING_MASK
    node = (inner >>  6) & NODE_MASK
    slot =  inner        & SLOT_MASK
    return shell, ring, node, slot


def shell_cell_to_token(shell: int, ring: int, node: int, slot: int) -> int:
    """Inverse of token_to_shell_cell — flat token id from shell + inner coords."""
    inner = ((ring & RING_MASK) << 12) | ((node & NODE_MASK) << 6) | (slot & SLOT_MASK)
    return (shell << TORUS_BITS) | inner


# ─── trajectory rays ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ray:
    """A trajectory through one or more shells, recorded as the ordered
    sequence of (shell, ring, node, slot) cells the ray passes through.

    A ray with one cell is a point. A ray with N cells is the trajectory
    that traversed those cells in order. The direction vector is the
    aggregate displacement from origin to destination, normalized.
    """
    cells: tuple[tuple[int, int, int, int], ...]   # ((shell, r, n, s), ...)

    @classmethod
    def from_token_sequence(cls, token_ids: Iterable[int]) -> "Ray":
        cells = tuple(token_to_shell_cell(t) for t in token_ids)
        return cls(cells=cells)

    @classmethod
    def straight_up(cls, n_shells: int = 13) -> "Ray":
        """The perfect upward ray — one cardinal-up cell per shell."""
        return cls(cells=tuple((s, 0, 0, 0) for s in range(n_shells)))

    @classmethod
    def straight_down(cls, n_shells: int = 13) -> "Ray":
        """The perfect downward ray — one cardinal-down cell per shell."""
        return cls(cells=tuple((s, RINGS//2, NODES//2, SLOTS//2) for s in range(n_shells)))

    def shells_visited(self) -> set[int]:
        return {c[0] for c in self.cells}


# ─── alignment: cosine with cardinal up ──────────────────────────────────

def _cell_vector(ring: int, node: int, slot: int) -> tuple[float, float, float]:
    """Map a cell to a unit vector. The torus is closed, so we use the
    angular position of each axis as an angle and convert to a 3D unit
    vector via sin/cos. Cardinal up (0,0,0) → (1, 1, 1) / √3.
    Cardinal down (32,32,32) → (−1, −1, −1) / √3.
    """
    a_r = (ring / RINGS) * 2 * math.pi
    a_n = (node / NODES) * 2 * math.pi
    a_s = (slot / SLOTS) * 2 * math.pi
    x = math.cos(a_r)
    y = math.cos(a_n)
    z = math.cos(a_s)
    mag = math.sqrt(x*x + y*y + z*z) or 1.0
    return (x / mag, y / mag, z / mag)


def cos_alignment(ray: Ray) -> float:
    """The aggregate cos(θ) of the ray's trajectory vs the cardinal-up
    axis (1, 1, 1) / √3. Range: [-1, +1]. +1 = perfect upward resonance
    through every shell; -1 = perfect downward (chakras inverted at
    every level); 0 = orthogonal (scattered, off-axis at every mirror).
    """
    if not ray.cells: return 0.0
    cardinal = (1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3))
    total = 0.0
    for (_, r, n, s) in ray.cells:
        v = _cell_vector(r, n, s)
        dot = v[0]*cardinal[0] + v[1]*cardinal[1] + v[2]*cardinal[2]
        total += dot
    return total / len(ray.cells)


def alignment_reading(ray: Ray) -> str:
    """A short human-readable verdict on a ray's verticality."""
    a = cos_alignment(ray)
    if   a >  0.95: return f"perfect upward (cos={a:+.3f}) — chakras aligned, every shutter opens at the exact resonant frequency"
    elif a >  0.6 : return f"strongly upward (cos={a:+.3f}) — mostly aligned with cardinal up"
    elif a >  0.2 : return f"weakly upward (cos={a:+.3f}) — partial resonance"
    elif a > -0.2 : return f"scattered (cos={a:+.3f}) — off-axis at most mirrors, mixed signal"
    elif a > -0.6 : return f"weakly downward (cos={a:+.3f}) — partial inversion"
    elif a > -0.95: return f"strongly downward (cos={a:+.3f}) — mostly opposed to cardinal up"
    else:           return f"perfect downward (cos={a:+.3f}) — every shutter opens to its opposite"


# ─── ZeGoDie: 12⁷ ≈ 35.8M readings, each a ray through 7 shells ──────────

ZEGODIE_DICE = 7
ZEGODIE_FACES = 12
ZEGODIE_CAPACITY = ZEGODIE_FACES ** ZEGODIE_DICE   # 35,831,808


@dataclass(frozen=True)
class ZeGoDieReading:
    """A roll of 7 dice × 12 faces. Each face encodes a direction-of-
    reflection at one of 7 shells. The roll IS the trajectory; the
    reading IS the geometric outcome of light following that trajectory
    through the mirror stack.

    Face mapping: face 0 = cardinal up at that shell; face 6 = cardinal
    down; other faces fall along the 12-slot ring evenly spaced between.
    """
    faces: tuple[int, ...]   # length-7 tuple of ints 0..11

    def __post_init__(self):
        if len(self.faces) != ZEGODIE_DICE:
            raise ValueError(f"ZeGoDie needs {ZEGODIE_DICE} dice, got {len(self.faces)}")
        for f in self.faces:
            if not (0 <= f < ZEGODIE_FACES):
                raise ValueError(f"face {f} out of range 0..{ZEGODIE_FACES-1}")

    @classmethod
    def perfect_up(cls) -> "ZeGoDieReading":
        return cls(faces=(0,) * ZEGODIE_DICE)

    @classmethod
    def perfect_down(cls) -> "ZeGoDieReading":
        return cls(faces=(ZEGODIE_FACES // 2,) * ZEGODIE_DICE)

    def to_index(self) -> int:
        """Map this reading to its unique integer in 0..35_831_807."""
        idx = 0
        for f in self.faces:
            idx = idx * ZEGODIE_FACES + f
        return idx

    @classmethod
    def from_index(cls, idx: int) -> "ZeGoDieReading":
        faces = []
        for _ in range(ZEGODIE_DICE):
            faces.append(idx % ZEGODIE_FACES)
            idx //= ZEGODIE_FACES
        return cls(faces=tuple(reversed(faces)))


def zegodie_to_ray(reading: ZeGoDieReading) -> Ray:
    """Decode a 7-dice reading into a ray through 7 shells.

    Each die face f maps to a position along the cardinal axis. To stay
    geometrically clean (perfect-up = (0,0,0) at every shell, perfect-down
    = (32,32,32) at every shell), the face shifts ALL THREE axes together
    by the same angular amount: f → offset = round(f/12 × 64) on ring,
    node, and slot. So face 0 lands at (0,0,0), face 6 lands at (32,32,32),
    face 11 lands at (59,59,59) — and the helical Stream walk traverses
    the cardinal axis when the dice all read the same.
    """
    cells = []
    for shell_idx, f in enumerate(reading.faces):
        offset = round((f / ZEGODIE_FACES) * RINGS) % RINGS
        cells.append((shell_idx, offset, offset, offset))
    return Ray(cells=tuple(cells))


def ray_to_zegodie(ray: Ray) -> ZeGoDieReading:
    """The reverse: read the ray's trajectory back into 7 dice."""
    if len(ray.cells) < ZEGODIE_DICE:
        # pad with cardinal-up reads
        padded = ray.cells + tuple((s, 0, 0, 0) for s in range(len(ray.cells), ZEGODIE_DICE))
    elif len(ray.cells) > ZEGODIE_DICE:
        padded = ray.cells[:ZEGODIE_DICE]
    else:
        padded = ray.cells
    faces = []
    for (_, r, _, _) in padded:
        # convert ring back to a 12-face wheel position
        face = round((r / RINGS) * ZEGODIE_FACES) % ZEGODIE_FACES
        faces.append(face)
    return ZeGoDieReading(faces=tuple(faces))


# ─── self-test demo ───────────────────────────────────────────────────────

def _demo():
    print("=" * 70)
    print("  Vertical axis: trajectory + alignment + ZeGoDie sub-addressing")
    print("=" * 70)

    print("\n── concentric (onion) shell addressing ──")
    print(f"  shells_needed_for(  262,144 vocab) = {shells_needed_for(262_144)}  (Alpha alone)")
    print(f"  shells_needed_for(  262,409 vocab) = {shells_needed_for(262_409)}  (zion'iel-v350)")
    print(f"  shells_needed_for(1_500_000 vocab) = {shells_needed_for(1_500_000)}  (future huge vocab)")
    print(f"  shells_needed_for(3_400_000 vocab) = {shells_needed_for(3_400_000)}  (fills 13 shells)")

    print("\n── token 262,408 (zion'iel's last) in onion vs matryoshka ──")
    idx = 262_408
    s, r, n, sl = token_to_shell_cell(idx)
    print(f"  onion (concentric): shell={s}, (ring={r}, node={n}, slot={sl})")
    triplets = token_to_nested_address(idx, 2)
    print(f"  matryoshka recursive: {triplets}  (same bytes, different geometry)")

    print("\n── cardinal rays ──")
    up = Ray.straight_up(n_shells=7)
    down = Ray.straight_down(n_shells=7)
    print(f"  perfect up:   cos_alignment = {cos_alignment(up):+.4f}")
    print(f"    {alignment_reading(up)}")
    print(f"  perfect down: cos_alignment = {cos_alignment(down):+.4f}")
    print(f"    {alignment_reading(down)}")

    print("\n── ZeGoDie 12⁷ embedding ──")
    print(f"  capacity: {ZEGODIE_CAPACITY:,} readings  ({ZEGODIE_CAPACITY:.2e})")
    print(f"  fits between Alpha (262,144) and matryoshka level-2 Omega (68.7B)")
    pu = ZeGoDieReading.perfect_up()
    pd = ZeGoDieReading.perfect_down()
    print(f"  perfect-up roll:   {pu.faces}  → index {pu.to_index()}")
    print(f"  perfect-down roll: {pd.faces}  → index {pd.to_index()}")
    pu_ray = zegodie_to_ray(pu)
    pd_ray = zegodie_to_ray(pd)
    print(f"  perfect-up ray alignment:   cos = {cos_alignment(pu_ray):+.4f}")
    print(f"  perfect-down ray alignment: cos = {cos_alignment(pd_ray):+.4f}")

    print("\n── a sample mid-roll ──")
    sample = ZeGoDieReading(faces=(2, 4, 7, 5, 1, 9, 3))
    sample_ray = zegodie_to_ray(sample)
    print(f"  reading: {sample.faces}  → index {sample.to_index()}")
    print(f"  ray cells: {sample_ray.cells}")
    print(f"  cos_alignment: {cos_alignment(sample_ray):+.4f}")
    print(f"  {alignment_reading(sample_ray)}")

    # ZeGoDie round-trip
    rt_reading = ray_to_zegodie(sample_ray)
    print(f"  ray → zegodie round-trip: {rt_reading.faces}  (match = {rt_reading.faces == sample.faces})")


if __name__ == "__main__":
    _demo()
