"""Unit tests for tokamld.imas — IMAS IDS equilibrium export and import.

Verifies:
1. Lossless round-trip of 2D field psi, grid R, Z (error <= 1e-12).
2. Lossless round-trip of 1D profiles: p', FF', q, p, F.
3. Lossless round-trip of LCFS outline and critical points (magnetic axis, X-points).
4. JSON export/import serialization without numerical precision loss.
5. Compliance with official ITER IMAS Data Dictionary path and node hierarchy.
"""
import sys
import tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))

import tokamld
from tokamld.imas import (
    EquilibriumData,
    to_ids,
    from_ids,
    export_equilibrium,
    import_equilibrium,
    save_ids_json,
    load_ids_json,
)


def make_sample_equilibrium() -> EquilibriumData:
    """Constructs a realistic Solov'ev equilibrium for testing."""
    nr, nz = 65, 65
    R = np.linspace(1.0, 2.4, nr, dtype=np.float64)
    Z = np.linspace(-1.2, 1.2, nz, dtype=np.float64)
    RR, ZZ = np.meshgrid(R, Z)

    # Analytic Solov'ev flux profile:
    # psi = (Z^2 / kappa^2) * (R^2 - R0^2) + (R^2 - R0^2)^2 / 8
    R0 = 1.7
    kappa = 1.6
    psi = (ZZ ** 2 / (kappa ** 2)) * (RR ** 2 - R0 ** 2) + ((RR ** 2 - R0 ** 2) ** 2) / 8.0

    # 1D profiles
    n_prof = 50
    psi_1d = np.linspace(float(np.min(psi)), float(np.max(psi)), n_prof, dtype=np.float64)
    pprime = -1.2e4 * np.ones(n_prof, dtype=np.float64)
    ffprime = -0.45 * np.ones(n_prof, dtype=np.float64)
    q = 1.1 + 2.4 * (np.linspace(0, 1, n_prof) ** 2)
    pressure = 1e5 * (1.0 - np.linspace(0, 1, n_prof) ** 2)
    f = 2.5 * np.ones(n_prof, dtype=np.float64)

    # LCFS outline (ellipse)
    theta = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
    lcfs_r = R0 + 0.5 * np.cos(theta)
    lcfs_z = 0.5 * kappa * np.sin(theta)

    # Critical points
    axis_r = float(R0)
    axis_z = 0.0
    x_points = [(1.3, -1.0), (1.3, 1.0)]

    return EquilibriumData(
        psi=psi,
        R=R,
        Z=Z,
        pprime=pprime,
        ffprime=ffprime,
        q=q,
        pressure=pressure,
        f=f,
        psi_1d=psi_1d,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
        axis_r=axis_r,
        axis_z=axis_z,
        psi_axis=float(np.min(psi)),
        psi_boundary=0.0,
        x_points=x_points,
        ip=1.5e6,
        time=1.25,
        shot=203702,
        boundary_type=1,
    )


def test_imas_roundtrip_lossless():
    """Test 1: Verify in-memory round-trip to_ids() -> from_ids() error <= 1e-12."""
    orig = make_sample_equilibrium()
    ids = to_ids(orig)
    rec = from_ids(ids)

    # 2D field and grid precision
    err_psi = np.max(np.abs(orig.psi - rec.psi))
    err_R = np.max(np.abs(orig.R - rec.R))
    err_Z = np.max(np.abs(orig.Z - rec.Z))

    assert err_psi <= 1e-12, f"Psi error exceeds tolerance: {err_psi}"
    assert err_R <= 1e-12, f"R grid error exceeds tolerance: {err_R}"
    assert err_Z <= 1e-12, f"Z grid error exceeds tolerance: {err_Z}"

    # 1D profiles
    assert np.max(np.abs(orig.pprime - rec.pprime)) <= 1e-12
    assert np.max(np.abs(orig.ffprime - rec.ffprime)) <= 1e-12
    assert np.max(np.abs(orig.q - rec.q)) <= 1e-12
    assert np.max(np.abs(orig.pressure - rec.pressure)) <= 1e-12
    assert np.max(np.abs(orig.f - rec.f)) <= 1e-12

    # Boundary and critical points
    assert np.max(np.abs(orig.lcfs_r - rec.lcfs_r)) <= 1e-12
    assert np.max(np.abs(orig.lcfs_z - rec.lcfs_z)) <= 1e-12
    assert abs(orig.axis_r - rec.axis_r) <= 1e-12
    assert abs(orig.axis_z - rec.axis_z) <= 1e-12
    assert abs(orig.psi_axis - rec.psi_axis) <= 1e-12
    assert abs(orig.psi_boundary - rec.psi_boundary) <= 1e-12
    assert abs(orig.ip - rec.ip) <= 1e-12
    assert abs(orig.time - rec.time) <= 1e-12
    assert orig.boundary_type == rec.boundary_type

    # X-points
    assert len(rec.x_points) == len(orig.x_points)
    for (xr_o, xz_o), (xr_r, xz_r) in zip(orig.x_points, rec.x_points):
        assert abs(xr_o - xr_r) <= 1e-12
        assert abs(xz_o - xz_r) <= 1e-12

    print("test_imas_roundtrip_lossless PASSED (error <= 1e-12)")


def test_imas_json_file_roundtrip():
    """Test 2: Verify export_equilibrium() -> import_equilibrium() JSON roundtrip."""
    orig = make_sample_equilibrium()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        export_equilibrium(orig, tmp_path)
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 1000

        rec = import_equilibrium(tmp_path)

        # Numerical equality
        assert np.max(np.abs(orig.psi - rec.psi)) <= 1e-12
        assert np.max(np.abs(orig.R - rec.R)) <= 1e-12
        assert np.max(np.abs(orig.Z - rec.Z)) <= 1e-12
        assert np.max(np.abs(orig.pprime - rec.pprime)) <= 1e-12
        assert np.max(np.abs(orig.ffprime - rec.ffprime)) <= 1e-12
        assert np.max(np.abs(orig.q - rec.q)) <= 1e-12
        assert abs(orig.axis_r - rec.axis_r) <= 1e-12
        assert abs(orig.axis_z - rec.axis_z) <= 1e-12
        print("test_imas_json_file_roundtrip PASSED")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_imas_schema_hierarchy_compliance():
    """Test 3: Verify exact IMAS Data Dictionary path compliance."""
    orig = make_sample_equilibrium()
    ids = to_ids(orig)

    # Check top-level properties
    assert "ids_properties" in ids
    assert ids["ids_properties"]["homogeneous_time"] == 1
    assert "time_slice" in ids
    assert len(ids["time_slice"]) == 1

    ts = ids["time_slice"][0]
    # Check 2D profile paths
    assert "profiles_2d" in ts
    p2d = ts["profiles_2d"][0]
    assert p2d["grid_type"]["index"] == 1
    assert p2d["grid_type"]["name"] == "rectangular"
    assert "dim1" in p2d["grid"]
    assert "dim2" in p2d["grid"]
    assert "psi" in p2d

    # Check 1D profile paths
    assert "profiles_1d" in ts
    p1d = ts["profiles_1d"]
    assert "dpressure_dpsi" in p1d
    assert "f_df_dpsi" in p1d
    assert "q" in p1d
    assert "pressure" in p1d
    assert "f" in p1d

    # Check boundary and global quantities paths
    assert "boundary" in ts
    assert "outline" in ts["boundary"]
    assert "r" in ts["boundary"]["outline"]
    assert "z" in ts["boundary"]["outline"]
    assert "x_point" in ts["boundary"]

    assert "global_quantities" in ts
    assert "magnetic_axis" in ts["global_quantities"]
    assert "r" in ts["global_quantities"]["magnetic_axis"]
    assert "z" in ts["global_quantities"]["magnetic_axis"]

    print("test_imas_schema_hierarchy_compliance PASSED")


def test_imas_dict_input():
    """Test 4: Verify to_ids accepts plain dictionary."""
    R = np.linspace(1.0, 2.0, 32)
    Z = np.linspace(-1.0, 1.0, 32)
    psi = np.random.randn(32, 32)
    raw_dict = {
        "psi": psi,
        "R": R,
        "Z": Z,
        "axis_r": 1.5,
        "axis_z": 0.0,
        "time": 0.5,
    }
    ids = to_ids(raw_dict)
    rec = from_ids(ids)
    assert np.max(np.abs(rec.psi - psi)) <= 1e-12
    assert rec.axis_r == 1.5
    print("test_imas_dict_input PASSED")


if __name__ == "__main__":
    test_imas_roundtrip_lossless()
    test_imas_json_file_roundtrip()
    test_imas_schema_hierarchy_compliance()
    test_imas_dict_input()
    print("\nALL IMAS UNIT TESTS PASSED!")
