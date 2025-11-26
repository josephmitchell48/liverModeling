import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, label, zoom
import vtk
from vtk.util import numpy_support


def load_binary_mask(path, label_value=None):
    img = nib.load(path)
    data = img.get_fdata()
    # If it's a label map, you can optionally pick a label_value
    if label_value is not None:
        mask = (data == label_value)
    else:
        # Assume nonzero voxels are "inside"
        mask = (data > 0)
    return mask.astype(np.uint8), img


def numpy_to_vtk_image(binary_vol, affine):
    """
    Convert a NumPy volume to vtkImageData.

    Assumes nibabel-style array shape: (X, Y, Z).

    We keep the axis order (X, Y, Z) and use Fortran ('F') flattening so that
    the X index is the fastest-varying, which is what VTK expects for
    vtkImageData with dimensions (nx, ny, nz).
    """
    # Ensure array is 3D and Fortran-contiguous
    vol = np.array(binary_vol, dtype=np.uint8, order="F")

    nx, ny, nz = vol.shape  # (X, Y, Z)

    # Voxel spacing from affine: columns correspond to x, y, z axes
    spacing_x = float(np.linalg.norm(affine[:3, 0]))
    spacing_y = float(np.linalg.norm(affine[:3, 1]))
    spacing_z = float(np.linalg.norm(affine[:3, 2]))

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(spacing_x, spacing_y, spacing_z)

    origin = affine[:3, 3]
    image.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))

    # Flatten in Fortran order so X is fastest-varying index
    flat = vol.ravel(order="F")
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=flat,
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    vtk_array.SetName("values")
    image.GetPointData().SetScalars(vtk_array)

    return image


def make_anisotropic_ball(radius_mm, spacing_xyz):
    """
    Build an ellipsoidal structuring element that approximates a ball in mm,
    accounting for anisotropic voxel spacing.
    """
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)

    rx, ry, rz = [max(1, int(np.ceil(radius_mm / s))) for s in spacing_xyz]
    xs = np.arange(-rx, rx + 1)
    ys = np.arange(-ry, ry + 1)
    zs = np.arange(-rz, rz + 1)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")

    ball = (xx / max(rx, 1)) ** 2 + (yy / max(ry, 1)) ** 2 + (zz / max(rz, 1)) ** 2 <= 1.0
    return ball.astype(bool)


def component_equiv_radius_mm(num_voxels, spacing_xyz):
    """
    Approximate equivalent sphere radius (mm) for a component of num_voxels.
    """
    if num_voxels <= 0:
        return 0.0
    voxel_vol = spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]
    vol = num_voxels * voxel_vol
    radius = ((3.0 * vol) / (4.0 * np.pi)) ** (1.0 / 3.0)
    return float(radius)


def make_hollow_shell(mask, spacing_xyz, wall_thickness_mm, guard_mm=0.0):
    """
    Hollow a binary mask by extracting its surface and giving it the requested wall thickness (mm).
    Surface-first avoids ballooning the structure away from the supplied vasculature geometry.
    """
    mask_bool = mask.astype(bool)
    # Use minimal erosion to extract a 1-voxel surface without changing geometry
    surface = mask_bool & (~binary_erosion(mask_bool, structure=np.ones((3, 3, 3), dtype=bool)))

    if wall_thickness_mm is None or wall_thickness_mm <= 0:
        shell = surface
    else:
        shell_struct = make_anisotropic_ball(wall_thickness_mm, spacing_xyz)
        # Grow surface outward only (do not re-dilate the whole volume) to keep fidelity
        shell = binary_dilation(surface, structure=shell_struct)

    if guard_mm and guard_mm > 0:
        # Closing to remove pinholes and make the tunnel shell continuous
        close_struct = make_anisotropic_ball(max(wall_thickness_mm, guard_mm, max(spacing_xyz)), spacing_xyz)
        shell = binary_erosion(binary_dilation(shell, structure=close_struct), structure=close_struct)
    return shell.astype(np.uint8)


def resample_mask_isotropic(mask, spacing_xyz, target_spacing_mm):
    """
    Resample a binary mask to isotropic spacing using trilinear interpolation,
    then threshold back to binary.
    """
    if target_spacing_mm is None or target_spacing_mm <= 0:
        return mask.astype(np.uint8), spacing_xyz
    zoom_factors = [spacing_xyz[0] / target_spacing_mm,
                    spacing_xyz[1] / target_spacing_mm,
                    spacing_xyz[2] / target_spacing_mm]
    resampled = zoom(mask.astype(np.float32), zoom=zoom_factors, order=1)
    return (resampled >= 0.5).astype(np.uint8), [target_spacing_mm] * 3


def marching_cubes_to_stl(vtk_image, iso_value, out_path, smoothing_iterations=0):
    # For binary data, vtkDiscreteMarchingCubes is often more robust
    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, iso_value)  # label value to extract
    mc.Update()

    # Optional smoothing to reduce noise; keep iterations modest to avoid shrinking thin features
    if smoothing_iterations > 0:
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(mc.GetOutputPort())
        smooth.SetNumberOfIterations(smoothing_iterations)
        smooth.SetRelaxationFactor(0.1)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOff()
        smooth.Update()
        output_port = smooth.GetOutputPort()
    else:
        output_port = mc.GetOutputPort()

    # Write STL
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(out_path)
    writer.SetInputConnection(output_port)
    writer.Write()
    print(f"[ok] Wrote STL: {out_path}")


def mask_to_polydata(mask_uint8, affine, iso_value=1):
    """
    Convert a binary mask to vtkPolyData via discrete marching cubes (no smoothing).
    """
    vtk_img = numpy_to_vtk_image(mask_uint8, affine)
    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, iso_value)
    mc.Update()
    return mc.GetOutput()


def fill_mesh_holes(polydata, hole_size_mm, spacing_xyz):
    """
    Fill mesh holes up to a given physical size (mm). This is mesh-only and does not
    alter the voxel tunnels; use a conservative hole_size_mm to avoid closing inlets.
    """
    if hole_size_mm is None or hole_size_mm <= 0:
        return polydata
    mean_spacing = float(np.mean(spacing_xyz))
    hole_size = hole_size_mm / mean_spacing if mean_spacing > 0 else hole_size_mm
    fh = vtk.vtkFillHolesFilter()
    fh.SetInputData(polydata)
    fh.SetHoleSize(hole_size)
    fh.Update()
    return fh.GetOutput()


def shift_mask(mask, shift_voxels):
    """
    Shift a mask by (dz, dy, dx) voxels without wraparound (pads with zeros).
    """
    dz, dy, dx = shift_voxels
    out = np.zeros_like(mask, dtype=mask.dtype)
    sz, sy, sx = mask.shape

    def slices(size, shift):
        if shift >= 0:
            src = slice(0, size - shift)
            dst = slice(shift, size)
        else:
            shift_abs = -shift
            src = slice(shift_abs, size)
            dst = slice(0, size - shift_abs)
        return src, dst

    src_z, dst_z = slices(sz, dz)
    src_y, dst_y = slices(sy, dy)
    src_x, dst_x = slices(sx, dx)

    out[dst_z, dst_y, dst_x] = mask[src_z, src_y, src_x]
    return out


def make_hollow_liver(
    liver_path,
    vessels_path,
    out_stl="hollow_liver.stl",
    vessel_dilation_mm=0.0,
    seal_margin_mm=0.0,
    vessel_shell_thickness_mm=0.0,
    vessel_shell_guard_mm=0.0,
    vessel_connectivity_fix_mm=0.0,
    vessel_min_radius_mm=None,
    vessel_small_dilation_mm=0.0,
    root_shell_thickness_mm=0.0,
    root_shell_out=None,
    separate_vessels=False,
    vessel_separation_step_mm=1.0,
    vessel_separation_max_steps=20,
    liver_pre_close_mm=0.0,
    liver_connectivity_fix_mm=0.0,
    liver_surface_close_mm=0.0,
    fill_surface_holes=False,
    hole_max_mm=0.0,
    keep_main_liver_component=True,
    vessel_label_a=None,
    vessel_label_b=None,
    vessel_raw_a_out=None,
    vessel_raw_b_out=None,
    op_iso_spacing_mm=0.5,
    vessel_raw_out=None,
    vessel_shell_out=None,
    vessel_shell_a_out=None,
    vessel_shell_b_out=None,
    liver_trimmed_out=None,
    filled_out=None,
    external_tunnel_thickness_mm=None,
    external_tunnel_out=None,
    external_tunnel_iso_spacing_mm=None,
    external_tunnel_merge_out=None,
    smoothing_iterations=0,
    liver_label=None,
    vessel_label=None,
):
    # Load masks
    liver_mask, liver_img = load_binary_mask(liver_path, liver_label)
    vessel_mask, vessel_img = load_binary_mask(vessels_path, vessel_label)

    # Basic sanity checks
    if liver_mask.shape != vessel_mask.shape:
        raise ValueError(f"Shape mismatch: liver {liver_mask.shape}, vessels {vessel_mask.shape}")
    if not np.allclose(liver_img.affine, vessel_img.affine):
        print("[warn] Liver and vessel affines differ slightly. Proceeding anyway.")

    print(f"[info] Volume shape: {liver_mask.shape}")

    vessel_data = vessel_img.get_fdata()
    vessel_unique = sorted(int(v) for v in np.unique(vessel_data) if v != 0)

    # Compute voxel spacing from affine
    base_affine = liver_img.affine
    spacing_x = float(np.linalg.norm(base_affine[:3, 0]))
    spacing_y = float(np.linalg.norm(base_affine[:3, 1]))
    spacing_z = float(np.linalg.norm(base_affine[:3, 2]))
    spacing_xyz = [spacing_x, spacing_y, spacing_z]
    spacing_orig = spacing_xyz[:]

    # Optional resample to isotropic grid to avoid stair-stepping (especially with thick Z spacing)
    if op_iso_spacing_mm and op_iso_spacing_mm > 0:
        liver_mask, _ = resample_mask_isotropic(liver_mask, spacing_orig, op_iso_spacing_mm)
        spacing_xyz = [op_iso_spacing_mm] * 3
        affine = np.eye(4)
        affine[0, 0] = spacing_xyz[0]
        affine[1, 1] = spacing_xyz[1]
        affine[2, 2] = spacing_xyz[2]
        affine[:3, 3] = base_affine[:3, 3]
        print(f"[info] Resampled liver to isotropic spacing ~{op_iso_spacing_mm} mm -> new shape {liver_mask.shape}")
    else:
        affine = base_affine

    # Optional pre-closing on liver mask to fill through-holes before subtraction (does not touch vessels/tunnels)
    if liver_pre_close_mm and liver_pre_close_mm > 0:
        pre_close_struct = make_anisotropic_ball(liver_pre_close_mm, spacing_xyz)
        liver_mask = binary_closing(liver_mask.astype(bool), structure=pre_close_struct).astype(np.uint8)

    # Build vessel masks for two networks (auto-detect labels if not provided)
    vessel_masks = []
    if vessel_label_a is not None:
        vessel_masks.append(("A", (vessel_data == vessel_label_a)))
    if vessel_label_b is not None:
        vessel_masks.append(("B", (vessel_data == vessel_label_b)))
    if not vessel_masks:
        if len(vessel_unique) >= 2:
            vessel_masks.append(("A", (vessel_data == vessel_unique[0])))
            vessel_masks.append(("B", (vessel_data == vessel_unique[1])))
        else:
            vessel_masks.append(("A", vessel_mask.astype(bool)))

    # Resample vessel masks if requested
    vessel_masks_resampled = []
    for name, vm in vessel_masks:
        vm_bool = vm.astype(bool)
        if op_iso_spacing_mm and op_iso_spacing_mm > 0:
            vm_iso, _ = resample_mask_isotropic(vm_bool.astype(np.uint8), spacing_orig, op_iso_spacing_mm)
            vessel_masks_resampled.append((name, vm_iso.astype(bool)))
        else:
            vessel_masks_resampled.append((name, vm_bool))

    # If requested, separate A/B to avoid contact by shifting B along +Z until no overlap (or max steps)
    if separate_vessels and len(vessel_masks_resampled) >= 2:
        # Expect first two to be A and B
        name_a, mask_a = vessel_masks_resampled[0]
        name_b, mask_b = vessel_masks_resampled[1]
        step_vox = max(1, int(round(vessel_separation_step_mm / spacing_xyz[2]))) if spacing_xyz[2] > 0 else 1
        shifted_b = mask_b
        total_shift = 0
        for _ in range(int(vessel_separation_max_steps)):
            overlap = mask_a & shifted_b
            if not overlap.any():
                break
            total_shift += step_vox
            shifted_b = shift_mask(mask_b, (0, 0, total_shift))
        vessel_masks_resampled[1] = (name_b, shifted_b)

    def process_single_vessel(name, vm_bool):
        mask = vm_bool.astype(bool)
        if vessel_dilation_mm > 0:
            structure = make_anisotropic_ball(vessel_dilation_mm, spacing_xyz)
            mask = binary_dilation(mask, structure=structure)
        if vessel_connectivity_fix_mm and vessel_connectivity_fix_mm > 0:
            fix_struct = make_anisotropic_ball(vessel_connectivity_fix_mm, spacing_xyz)
            mask = binary_erosion(binary_dilation(mask, structure=fix_struct), structure=fix_struct)

        labeled, num_components = label(mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=int))
        outside_ids = []
        outside_counts = {}
        for comp_id in range(1, num_components + 1):
            comp_mask = labeled == comp_id
            if np.any(comp_mask & (~liver_mask.astype(bool))):
                outside_ids.append(comp_id)
                outside_counts[comp_id] = int(np.count_nonzero(comp_mask))

        keep_id = None
        if outside_ids:
            keep_id = max(outside_ids, key=lambda cid: outside_counts[cid])
            if len(vessel_masks_resampled) == 1:
                print(f"[info] ({name}) Keeping external component id={keep_id} open; sealing {len(outside_ids) - 1} others.")
        else:
            if len(vessel_masks_resampled) == 1:
                print(f"[info] ({name}) No vessel component extends outside the liver; nothing to seal.")

        seal_structure = make_anisotropic_ball(seal_margin_mm, spacing_xyz) if seal_margin_mm > 0 else np.ones((1, 1, 1), dtype=bool)
        adjusted_vessels = np.zeros_like(mask, dtype=bool)

        for comp_id in range(1, num_components + 1):
            comp_mask = labeled == comp_id
            if comp_id == keep_id:
                adjusted = comp_mask
            elif keep_id is not None and comp_id in outside_ids:
                eroded = binary_erosion(comp_mask, structure=seal_structure)
                adjusted = eroded & liver_mask.astype(bool)
            else:
                adjusted = comp_mask
            adjusted_vessels |= adjusted

        adjusted_uint8 = adjusted_vessels.astype(np.uint8)

        # Optional conditional dilation for small components
        if vessel_min_radius_mm and vessel_min_radius_mm > 0 and vessel_small_dilation_mm and vessel_small_dilation_mm > 0:
            dil_struct = make_anisotropic_ball(vessel_small_dilation_mm, spacing_xyz)
            labeled_adj, num_adj = label(adjusted_uint8, structure=np.ones((3, 3, 3), dtype=int))
            conditioned = np.zeros_like(adjusted_uint8, dtype=bool)
            for cid in range(1, num_adj + 1):
                comp = labeled_adj == cid
                radius_mm = component_equiv_radius_mm(int(comp.sum()), spacing_xyz)
                if radius_mm < vessel_min_radius_mm:
                    comp = binary_dilation(comp, structure=dil_struct)
                conditioned |= comp
            adjusted_uint8 = conditioned.astype(np.uint8)

        return adjusted_uint8, keep_id

    processed_masks = []
    processed_shells = []

    for name, vm in vessel_masks_resampled:
        adjusted_mask, keep_id = process_single_vessel(name, vm)

        if vessel_raw_out and name == "A":
            vtk_raw = numpy_to_vtk_image(adjusted_mask, affine)
            marching_cubes_to_stl(vtk_raw, iso_value=1, out_path=vessel_raw_out, smoothing_iterations=smoothing_iterations)
        if vessel_raw_a_out and name == "A":
            vtk_raw = numpy_to_vtk_image(adjusted_mask, affine)
            marching_cubes_to_stl(vtk_raw, iso_value=1, out_path=vessel_raw_a_out, smoothing_iterations=smoothing_iterations)
        if vessel_raw_b_out and name == "B":
            vtk_raw = numpy_to_vtk_image(adjusted_mask, affine)
            marching_cubes_to_stl(vtk_raw, iso_value=1, out_path=vessel_raw_b_out, smoothing_iterations=smoothing_iterations)

        vessel_shell = make_hollow_shell(adjusted_mask, spacing_xyz, vessel_shell_thickness_mm, guard_mm=vessel_shell_guard_mm)

        if vessel_shell_a_out and name == "A":
            vtk_shell_a = numpy_to_vtk_image(vessel_shell, affine)
            marching_cubes_to_stl(vtk_shell_a, iso_value=1, out_path=vessel_shell_a_out, smoothing_iterations=smoothing_iterations)
        if vessel_shell_b_out and name == "B":
            vtk_shell_b = numpy_to_vtk_image(vessel_shell, affine)
            marching_cubes_to_stl(vtk_shell_b, iso_value=1, out_path=vessel_shell_b_out, smoothing_iterations=smoothing_iterations)

        processed_masks.append(adjusted_mask.astype(bool))
        processed_shells.append(vessel_shell.astype(bool))
        print(f"[info] ({name}) mask voxels: {int(adjusted_mask.sum())}, shell voxels: {int(vessel_shell.sum())}")

    vessel_mask_union = np.zeros_like(liver_mask, dtype=bool)
    for pm in processed_masks:
        vessel_mask_union |= pm
    shell_union = np.zeros_like(liver_mask, dtype=bool)
    for sh in processed_shells:
        shell_union |= sh

    print(f"[info] Shell union voxels: {int(shell_union.sum())}")
    if vessel_shell_out:
        if len(processed_shells) >= 2:
            # Append shell meshes so both A and B always appear even if a viewer hides thin overlap
            append = vtk.vtkAppendPolyData()
            for sh in processed_shells:
                pd = mask_to_polydata(sh.astype(np.uint8), affine, iso_value=1)
                append.AddInputData(pd)
            append.Update()
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(vessel_shell_out)
            writer.SetInputData(append.GetOutput())
            writer.Write()
            print(f"[ok] Wrote STL (appended shells): {vessel_shell_out}")
        else:
            vtk_shell = numpy_to_vtk_image(shell_union.astype(np.uint8), affine)
            marching_cubes_to_stl(vtk_shell, iso_value=1, out_path=vessel_shell_out, smoothing_iterations=smoothing_iterations)

    # Remove vasculature volume from liver (keep vessels intact; liver is carved)
    liver_trimmed = liver_mask.astype(bool) & (~vessel_mask_union.astype(bool))

    # Optional closing to patch small discontinuities in the liver shell
    if liver_connectivity_fix_mm and liver_connectivity_fix_mm > 0:
        liver_fix_struct = make_anisotropic_ball(liver_connectivity_fix_mm, spacing_xyz)
        liver_trimmed = binary_closing(liver_trimmed, structure=liver_fix_struct)

    # Optionally keep only the largest liver component to remove crumbs
    if keep_main_liver_component:
        labeled_liver, num_liver = label(liver_trimmed.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=int))
        if num_liver > 1:
            counts = [(i, int(np.count_nonzero(labeled_liver == i))) for i in range(1, num_liver + 1)]
            main_id = max(counts, key=lambda x: x[1])[0]
            liver_trimmed = labeled_liver == main_id

    liver_trimmed = liver_trimmed.astype(np.uint8)

    print(f"[info] Vessel voxels (union): {vessel_mask_union.sum()} → shell voxels (union): {shell_union.sum()}")
    print(f"[info] Liver voxels before: {liver_mask.sum()}, after carving+fix: {liver_trimmed.sum()}")

    # Optional root shell: hollow the main external component (root) with thickness, keep inlet open
    root_shell = np.zeros_like(liver_mask, dtype=bool)
    if root_shell_thickness_mm and root_shell_thickness_mm > 0:
        labeled_union, num_components_union = label(vessel_mask_union.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=int))
        outside_ids = []
        outside_counts = {}
        for comp_id in range(1, num_components_union + 1):
            comp_mask = labeled_union == comp_id
            if np.any(comp_mask & (~liver_mask.astype(bool))):
                outside_ids.append(comp_id)
                outside_counts[comp_id] = int(np.count_nonzero(comp_mask))
        root_id = max(outside_ids, key=lambda cid: outside_counts[cid]) if outside_ids else None
        if root_id is not None:
            root_mask = (labeled_union == root_id)
            root_shell = make_hollow_shell(root_mask.astype(np.uint8), spacing_xyz, root_shell_thickness_mm, guard_mm=vessel_shell_guard_mm).astype(bool)
            if root_shell_out:
                vtk_root = numpy_to_vtk_image(root_shell.astype(np.uint8), affine)
                marching_cubes_to_stl(vtk_root, iso_value=1, out_path=root_shell_out, smoothing_iterations=smoothing_iterations)

    # Union of carved liver + hollow vasculature (trunk/root stays)
    merged_mask = (liver_trimmed.astype(bool) | shell_union.astype(bool) | root_shell.astype(bool)).astype(np.uint8)

    # Convert to VTK images for meshing
    vtk_merged = numpy_to_vtk_image(merged_mask, affine)
    marching_cubes_to_stl(vtk_merged, iso_value=1, out_path=out_stl, smoothing_iterations=smoothing_iterations)

    # Optional: mesh-only hole filling on merged mesh, writing to filled_out to preserve tunnels
    if fill_surface_holes and filled_out:
        merged_poly = mask_to_polydata(merged_mask.astype(np.uint8), affine, iso_value=1)
        filled_poly = fill_mesh_holes(merged_poly, hole_max_mm, spacing_xyz)
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(filled_out)
        writer.SetInputData(filled_poly)
        writer.Write()
        print(f"[ok] Wrote hole-filled STL: {filled_out}")

    if liver_trimmed_out:
        vtk_liver = numpy_to_vtk_image(liver_trimmed, affine)
        marching_cubes_to_stl(vtk_liver, iso_value=1, out_path=liver_trimmed_out, smoothing_iterations=smoothing_iterations)

    if vessel_shell_out:
        vtk_shell = numpy_to_vtk_image(vessel_shell, affine)
        marching_cubes_to_stl(vtk_shell, iso_value=1, out_path=vessel_shell_out, smoothing_iterations=smoothing_iterations)

    # Optional: export a thin tunnel around the external segment outside the liver
    if external_tunnel_thickness_mm and external_tunnel_thickness_mm > 0:
        labeled_union, num_components_union = label(vessel_mask_union.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=int))
        outside_ids = []
        outside_counts = {}
        for comp_id in range(1, num_components_union + 1):
            comp_mask = labeled_union == comp_id
            if np.any(comp_mask & (~liver_mask.astype(bool))):
                outside_ids.append(comp_id)
                outside_counts[comp_id] = int(np.count_nonzero(comp_mask))

        keep_id = max(outside_ids, key=lambda cid: outside_counts[cid]) if outside_ids else None
        if keep_id is None:
            print("[info] No external component to create a tunnel from.")
        else:
            external_outside = (labeled_union == keep_id) & (~liver_mask.astype(bool))
            if not external_outside.any():
                print("[info] External component does not extend outside liver; no tunnel created.")
            else:
                tunnel_mask = external_outside.astype(np.uint8)
                tunnel_spacing = spacing_xyz
                if external_tunnel_iso_spacing_mm and external_tunnel_iso_spacing_mm > 0:
                    tunnel_mask, tunnel_spacing = resample_mask_isotropic(
                        tunnel_mask, spacing_xyz, external_tunnel_iso_spacing_mm
                    )
                    tunnel_affine = np.eye(4)
                    tunnel_affine[0, 0] = tunnel_spacing[0]
                    tunnel_affine[1, 1] = tunnel_spacing[1]
                    tunnel_affine[2, 2] = tunnel_spacing[2]
                    tunnel_affine[:3, 3] = affine[:3, 3]
                else:
                    tunnel_affine = affine

                shell_struct = make_anisotropic_ball(external_tunnel_thickness_mm, tunnel_spacing)
                dilated = binary_dilation(tunnel_mask.astype(bool), structure=shell_struct)
                shell = dilated & (~tunnel_mask.astype(bool))
                close_struct = make_anisotropic_ball(max(external_tunnel_thickness_mm, tunnel_spacing[0]), tunnel_spacing)
                shell_closed = binary_erosion(binary_dilation(shell, structure=close_struct), structure=close_struct)
                shell_mask = shell_closed.astype(np.uint8)
                tunnel_path = external_tunnel_out or out_stl.replace(".stl", "_external_tunnel.stl")
                vtk_shell = numpy_to_vtk_image(shell_mask, tunnel_affine)
                marching_cubes_to_stl(vtk_shell, iso_value=1, out_path=tunnel_path, smoothing_iterations=smoothing_iterations)
                print(f"[ok] Wrote external tunnel STL: {tunnel_path}")

                merge_path = external_tunnel_merge_out or out_stl.replace(".stl", "_with_tunnel.stl")
                if external_tunnel_iso_spacing_mm and external_tunnel_iso_spacing_mm > 0:
                    merged_affine = tunnel_affine
                    liver_resampled, _ = resample_mask_isotropic(liver_trimmed, spacing_xyz, tunnel_spacing[0])
                    shell_union_resampled, _ = resample_mask_isotropic(shell_union.astype(np.uint8), spacing_xyz, tunnel_spacing[0])
                else:
                    merged_affine = affine
                    liver_resampled = liver_trimmed
                    shell_union_resampled = shell_union
                union_mask = (liver_resampled.astype(bool) | shell_mask.astype(bool) | shell_union_resampled.astype(bool)).astype(np.uint8)
                vtk_union = numpy_to_vtk_image(union_mask, merged_affine)
                marching_cubes_to_stl(vtk_union, iso_value=1, out_path=merge_path, smoothing_iterations=smoothing_iterations)
                print(f"[ok] Wrote merged liver+tunnel STL: {merge_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a hollow liver-with-trunk STL: hollow the vasculature, carve the liver, and keep the trunk shell."
    )
    parser.add_argument("liver_nii", help="Path to liver mask NIfTI (e.g. liver.nii.gz)")
    parser.add_argument("vessels_nii", help="Path to vessel mask NIfTI (e.g. vessels.nii.gz)")
    parser.add_argument("--out", default="hollow_liver.stl", help="Output STL file name")
    parser.add_argument("--vessel-dilation-mm", type=float, default=0.0,
                        help="Approximate vessel dilation in mm before subtraction (default: 0.0 = off)")
    parser.add_argument("--seal-margin-mm", type=float, default=0.0,
                        help="Margin (mm) to erode boundary-touching vessels, except the main external trunk (default: 0.0 = no change)")
    parser.add_argument("--vessel-shell-thickness-mm", type=float, default=0.0,
                        help="Wall thickness (mm) for hollowing the entire vasculature (including trunk) into a tunnel (default: 0 = 1-voxel surface)")
    parser.add_argument("--vessel-shell-guard-mm", type=float, default=0.0,
                        help="Closing kernel (mm) to remove pinholes in the vasculature shell (default 0; set >0 to enable)")
    parser.add_argument("--vessel-connectivity-fix-mm", type=float, default=0.0,
                        help="Closing kernel (mm) to repair tiny gaps in vessel mask before shelling (default 0; set >0 to enable)")
    parser.add_argument("--root-shell-thickness-mm", type=float, default=0.0,
                        help="Wall thickness (mm) to hollow the main external root/trunk (default 0 = off)")
    parser.add_argument("--root-shell-out", type=str, default=None,
                        help="Optional STL path for root shell (default: skip)")
    parser.add_argument("--vessel-min-radius-mm", type=float, default=None,
                        help="If set, dilate vessel components smaller than this radius (mm) by --vessel-small-dilation-mm")
    parser.add_argument("--vessel-small-dilation-mm", type=float, default=0.0,
                        help="Dilation (mm) applied to vessel components smaller than --vessel-min-radius-mm")
    parser.add_argument("--liver-pre-close-mm", type=float, default=0.0,
                        help="Closing kernel (mm) applied to the liver mask before subtraction to fill through-holes (default 0)")
    parser.add_argument("--separate-vessels", action="store_true", default=False,
                        help="Shift vessel B along +Z until it no longer overlaps vessel A (default off)")
    parser.add_argument("--vessel-separation-step-mm", type=float, default=1.0,
                        help="Step size (mm) when separating vessels if --separate-vessels is enabled (default 1.0)")
    parser.add_argument("--vessel-separation-max-steps", type=int, default=20,
                        help="Max steps when separating vessels if --separate-vessels is enabled (default 20)")
    parser.add_argument("--liver-connectivity-fix-mm", type=float, default=1.0,
                        help="Closing kernel (mm) to patch small discontinuities in carved liver (default 1.0; set 0 to disable)")
    parser.add_argument("--fill-surface-holes", action="store_true", default=False,
                        help="Apply mesh-only hole filling on merged mesh (outer surface; tunnels preserved)")
    parser.add_argument("--hole-max-mm", type=float, default=0.0,
                        help="Max hole size (mm) to fill when --fill-surface-holes is enabled (default 0 = off)")
    parser.add_argument("--keep-main-liver-component", action="store_true", default=True,
                        help="Keep only the largest liver component after carving/closing (default on)")
    parser.add_argument("--op-iso-spacing-mm", type=float, default=0.5,
                        help="Optional isotropic resample spacing for operations to reduce stair-step artifacts (default 0.5; set <=0 to disable)")
    parser.add_argument("--vessel-raw-out", type=str, default=None,
                        help="Optional STL path for the raw vessel mask (diagnostic)")
    parser.add_argument("--vessel-raw-a-out", type=str, default=None,
                        help="Optional STL path for raw vessel mask A (diagnostic)")
    parser.add_argument("--vessel-raw-b-out", type=str, default=None,
                        help="Optional STL path for raw vessel mask B (diagnostic)")
    parser.add_argument("--vessel-shell-out", type=str, default=None,
                        help="Optional STL path for the hollow vasculature shell (default: skip)")
    parser.add_argument("--vessel-shell-a-out", type=str, default=None,
                        help="Optional STL path for vessel shell A (default: skip)")
    parser.add_argument("--vessel-shell-b-out", type=str, default=None,
                        help="Optional STL path for vessel shell B (default: skip)")
    parser.add_argument("--liver-trimmed-out", type=str, default=None,
                        help="Optional STL path for the liver after carving out the vasculature (default: skip)")
    parser.add_argument("--filled-out", type=str, default=None,
                        help="Optional STL path for the hole-filled mesh (mesh-only fill)")
    parser.add_argument("--external-tunnel-thickness-mm", type=float, default=None,
                        help="If set, create a thin tunnel around the external trunk outside the liver with this wall thickness (mm)")
    parser.add_argument("--external-tunnel-out", type=str, default=None,
                        help="Output STL path for external tunnel (default: based on --out)")
    parser.add_argument("--external-tunnel-iso-spacing-mm", type=float, default=None,
                        help="If set, resample the external tunnel mask to isotropic spacing before shelling (e.g. 1.0)")
    parser.add_argument("--external-tunnel-merge-out", type=str, default=None,
                        help="Output STL path for merged liver+tunnel (default: based on --out)")
    parser.add_argument("--smoothing-iterations", type=int, default=0,
                        help="Optional smoothing iterations on mesh (default: 0 to avoid shrinking outlets)")
    parser.add_argument("--liver-label", type=int, default=None,
                        help="If liver NIfTI is a label map, specify liver label value")
    parser.add_argument("--vessel-label", type=int, default=None,
                        help="If vessel NIfTI is a label map, specify vessel label value")
    parser.add_argument("--vessel-label-a", type=int, default=None,
                        help="Label value for vessel network A (if the vessel NIfTI has multiple labels)")
    parser.add_argument("--vessel-label-b", type=int, default=None,
                        help="Label value for vessel network B (if the vessel NIfTI has multiple labels)")

    args = parser.parse_args()

    make_hollow_liver(
        liver_path=args.liver_nii,
        vessels_path=args.vessels_nii,
        out_stl=args.out,
        vessel_dilation_mm=args.vessel_dilation_mm,
        seal_margin_mm=args.seal_margin_mm,
        vessel_shell_thickness_mm=args.vessel_shell_thickness_mm,
        vessel_shell_guard_mm=args.vessel_shell_guard_mm,
        vessel_connectivity_fix_mm=args.vessel_connectivity_fix_mm,
        liver_pre_close_mm=args.liver_pre_close_mm,
        vessel_min_radius_mm=args.vessel_min_radius_mm,
        vessel_small_dilation_mm=args.vessel_small_dilation_mm,
        root_shell_thickness_mm=args.root_shell_thickness_mm,
        root_shell_out=args.root_shell_out,
        fill_surface_holes=args.fill_surface_holes,
        hole_max_mm=args.hole_max_mm,
        separate_vessels=args.separate_vessels,
        vessel_separation_step_mm=args.vessel_separation_step_mm,
        vessel_separation_max_steps=args.vessel_separation_max_steps,
        liver_connectivity_fix_mm=args.liver_connectivity_fix_mm,
        keep_main_liver_component=args.keep_main_liver_component,
        op_iso_spacing_mm=args.op_iso_spacing_mm,
        vessel_raw_out=args.vessel_raw_out,
        vessel_raw_a_out=args.vessel_raw_a_out,
        vessel_raw_b_out=args.vessel_raw_b_out,
        vessel_shell_out=args.vessel_shell_out,
        vessel_shell_a_out=args.vessel_shell_a_out,
        vessel_shell_b_out=args.vessel_shell_b_out,
        liver_trimmed_out=args.liver_trimmed_out,
        filled_out=getattr(args, "filled_out", None),
        external_tunnel_thickness_mm=args.external_tunnel_thickness_mm,
        external_tunnel_out=args.external_tunnel_out,
        external_tunnel_iso_spacing_mm=args.external_tunnel_iso_spacing_mm,
        external_tunnel_merge_out=args.external_tunnel_merge_out,
        smoothing_iterations=args.smoothing_iterations,
        liver_label=args.liver_label,
        vessel_label=args.vessel_label,
        vessel_label_a=args.vessel_label_a,
        vessel_label_b=args.vessel_label_b,
    )
