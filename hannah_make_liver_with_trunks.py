import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, zoom, label
import vtk
from vtk.util import numpy_support


def load_binary_mask(path, label_value=None):
    img = nib.load(path)
    data = img.get_fdata()
    if label_value is not None:
        mask = data == label_value
    else:
        mask = data > 0
    return mask.astype(bool), img


def make_anisotropic_ball(radius_mm, spacing_xyz):
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    rx, ry, rz = [max(1, int(np.ceil(radius_mm / s))) for s in spacing_xyz]
    xs = np.arange(-rx, rx + 1)
    ys = np.arange(-ry, ry + 1)
    zs = np.arange(-rz, rz + 1)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    ball = (xx / max(rx, 1)) ** 2 + (yy / max(ry, 1)) ** 2 + (zz / max(rz, 1)) ** 2 <= 1.0
    return ball.astype(bool)


def resample_mask_iso(mask, spacing_xyz, target_spacing_mm):
    zoom_factors = [spacing_xyz[0] / target_spacing_mm,
                    spacing_xyz[1] / target_spacing_mm,
                    spacing_xyz[2] / target_spacing_mm]
    resampled = zoom(mask.astype(np.float32), zoom=zoom_factors, order=1)
    return (resampled >= 0.5)


def write_mask_to_stl(mask, spacing_mm, out_path, smoothing_iterations=0):
    vol = np.array(mask, dtype=np.uint8, order="F")
    nx, ny, nz = vol.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(spacing_mm, spacing_mm, spacing_mm)
    image.SetOrigin(0.0, 0.0, 0.0)
    flat = vol.ravel(order="F")
    arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    arr.SetName("values")
    image.GetPointData().SetScalars(arr)

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(image)
    mc.SetValue(0, 1)
    mc.Update()

    output_port = mc.GetOutputPort()
    if smoothing_iterations and smoothing_iterations > 0:
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(mc.GetOutputPort())
        smooth.SetNumberOfIterations(int(smoothing_iterations))
        smooth.SetRelaxationFactor(0.1)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOff()
        smooth.Update()
        output_port = smooth.GetOutputPort()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(out_path)
    writer.SetInputConnection(output_port)
    writer.Write()
    print(f"[ok] Wrote {out_path}")


def process_vessel_mask(vessel_mask, liver_mask, spacing_xyz, seal_margin_mm):
    labeled, num_components = label(vessel_mask, structure=np.ones((3, 3, 3), dtype=int))
    outside_ids = []
    outside_counts = {}
    for comp_id in range(1, num_components + 1):
        comp_mask = labeled == comp_id
        if np.any(comp_mask & (~liver_mask)):
            outside_ids.append(comp_id)
            outside_counts[comp_id] = int(comp_mask.sum())

    keep_id = max(outside_ids, key=lambda cid: outside_counts[cid]) if outside_ids else None
    if seal_margin_mm > 0:
        seal_struct = make_anisotropic_ball(seal_margin_mm, spacing_xyz)
    else:
        seal_struct = np.ones((1, 1, 1), dtype=bool)

    adjusted = np.zeros_like(vessel_mask, dtype=bool)
    for comp_id in range(1, num_components + 1):
        comp_mask = labeled == comp_id
        if comp_id == keep_id:
            adjusted |= comp_mask
        elif keep_id is not None and comp_id in outside_ids:
            # Seal boundary-touching vessels away from surface while keeping internal branches
            eroded = binary_erosion(comp_mask, structure=seal_struct)
            adjusted |= (eroded & liver_mask)
        else:
            adjusted |= comp_mask

    external_outside = (labeled == keep_id) & (~liver_mask) if keep_id is not None else np.zeros_like(vessel_mask, dtype=bool)
    return adjusted, external_outside


def build_shell(external_mask, spacing_xyz, thickness_mm, guard_mm=0.0):
    if thickness_mm <= 0 or not external_mask.any():
        return np.zeros_like(external_mask, dtype=bool)
    shell_struct = make_anisotropic_ball(thickness_mm, spacing_xyz)
    dilated = binary_dilation(external_mask, structure=shell_struct)
    shell = dilated & (~external_mask)
    close_struct = make_anisotropic_ball(max(thickness_mm, max(spacing_xyz)), spacing_xyz)
    shell_closed = binary_erosion(binary_dilation(shell, structure=close_struct), structure=close_struct)
    if guard_mm and guard_mm > 0:
        # Extra closing pass with larger kernel to ensure no pinholes in the trunk shell
        guard_struct = make_anisotropic_ball(thickness_mm + guard_mm, spacing_xyz)
        shell_closed = binary_erosion(binary_dilation(shell_closed, structure=guard_struct), structure=guard_struct)
    return shell_closed.astype(bool)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build hollow liver with separate portal/hepatic tunnels and external trunks (Hannah workflow).")
    parser.add_argument("liver_nii", help="Liver mask NIfTI")
    parser.add_argument("portal_nii", help="Portal mask NIfTI (if only one vessel mask, pass it here and leave hepatic empty)")
    parser.add_argument("hepatic_nii", nargs="?", default=None, help="Hepatic mask NIfTI (optional; defaults to portal mask if omitted)")
    parser.add_argument("--out-prefix", default=None, help="Prefix for output STLs (defaults to --out basename or 'hannah_case')")
    parser.add_argument("--out", default=None, help="Optional merged liver_with_trunks STL path (if set, also write this single file)")
    parser.add_argument("--iso-spacing-mm", type=float, default=0.5, help="Isotropic spacing for union/remesh (default 0.5mm; set <=0 to stay native grid)")
    parser.add_argument("--seal-margin-mm", type=float, default=0.5, help="Erode non-trunk boundary-touching vessels by this margin (mm)")
    parser.add_argument("--shell-thickness-mm", type=float, default=3.0, help="Shell thickness for external trunks (mm)")
    parser.add_argument("--trunk-guard-mm", type=float, default=0.0, help="Extra closing kernel to eliminate pinholes in trunk shell (mm; 0 = off)")
    parser.add_argument("--vessel-dilation-mm", type=float, default=0.0, help="Optional global vessel dilation before processing (default 0.0)")
    parser.add_argument("--connectivity-fix-mm", type=float, default=0.5, help="Closing radius to fix tiny gaps in vessel mask (default 0.5mm; set 0 to disable)")
    parser.add_argument("--smoothing-iterations", type=int, default=5, help="Mesh smoothing iterations (default 5)")
    parser.add_argument("--pad-voxels", type=int, default=4, help="Padding voxels around crop before resampling (default 4)")
    parser.add_argument("--liver-label", type=int, default=None, help="Label value for liver mask (if labeled)")
    parser.add_argument("--portal-label", type=int, default=None, help="Label value for portal mask (if labeled)")
    parser.add_argument("--hepatic-label", type=int, default=None, help="Label value for hepatic mask (if labeled)")
    args = parser.parse_args()

    liver_mask, liver_img = load_binary_mask(args.liver_nii, args.liver_label)
    portal_mask, portal_img = load_binary_mask(args.portal_nii, args.portal_label)
    if args.hepatic_nii is not None:
        hepatic_mask, hepatic_img = load_binary_mask(args.hepatic_nii, args.hepatic_label)
    else:
        hepatic_mask, hepatic_img = portal_mask, portal_img

    if liver_mask.shape != portal_mask.shape or liver_mask.shape != hepatic_mask.shape:
        raise ValueError("Mask shapes do not match.")
    spacing_xyz = [
        float(np.linalg.norm(liver_img.affine[:3, 0])),
        float(np.linalg.norm(liver_img.affine[:3, 1])),
        float(np.linalg.norm(liver_img.affine[:3, 2])),
    ]

    # Optional global dilation on vessels
    if args.vessel_dilation_mm > 0:
        struct = make_anisotropic_ball(args.vessel_dilation_mm, spacing_xyz)
        portal_mask = binary_dilation(portal_mask, structure=struct)
        hepatic_mask = binary_dilation(hepatic_mask, structure=struct)

    # Connectivity fix: close tiny gaps to preserve continuity
    if args.connectivity_fix_mm > 0:
        fix_struct = make_anisotropic_ball(args.connectivity_fix_mm, spacing_xyz)
        portal_mask = binary_erosion(binary_dilation(portal_mask, structure=fix_struct), structure=fix_struct)
        hepatic_mask = binary_erosion(binary_dilation(hepatic_mask, structure=fix_struct), structure=fix_struct)

    portal_proc, portal_external = process_vessel_mask(portal_mask, liver_mask, spacing_xyz, args.seal_margin_mm)
    hepatic_proc, hepatic_external = process_vessel_mask(hepatic_mask, liver_mask, spacing_xyz, args.seal_margin_mm)

    vessels_combined = portal_proc | hepatic_proc
    liver_hollow = liver_mask & (~vessels_combined)

    portal_shell = build_shell(portal_external, spacing_xyz, args.shell_thickness_mm, guard_mm=args.trunk_guard_mm)
    hepatic_shell = build_shell(hepatic_external, spacing_xyz, args.shell_thickness_mm, guard_mm=args.trunk_guard_mm)
    shells_combined = portal_shell | hepatic_shell

    # Crop to bounding box of everything to keep resample manageable
    total_mask_native = liver_hollow | shells_combined
    coords = np.argwhere(total_mask_native)
    if coords.size == 0:
        raise ValueError("No voxels in combined mask; check inputs.")
    mins = coords.min(axis=0) - args.pad_voxels
    maxs = coords.max(axis=0) + args.pad_voxels
    mins = np.maximum(mins, 0)
    maxs = np.minimum(maxs, np.array(total_mask_native.shape) - 1)
    slc = tuple(slice(mn, mx + 1) for mn, mx in zip(mins, maxs))

    liver_crop = liver_hollow[slc]
    portal_shell_crop = portal_shell[slc]
    hepatic_shell_crop = hepatic_shell[slc]

    iso_spacing = args.iso_spacing_mm
    # Derive prefix if not provided
    if args.out_prefix:
        prefix = args.out_prefix
    elif args.out:
        prefix = args.out.rsplit(".", 1)[0]
    else:
        prefix = "hannah_case"
    liver_only_path = f"{prefix}_liver_only.stl"
    shells_only_path = f"{prefix}_shells_only.stl"
    merged_path = f"{prefix}_liver_with_trunks.stl"

    if iso_spacing and iso_spacing > 0:
        liver_iso = resample_mask_iso(liver_crop, spacing_xyz, iso_spacing)
        portal_shell_iso = resample_mask_iso(portal_shell_crop, spacing_xyz, iso_spacing)
        hepatic_shell_iso = resample_mask_iso(hepatic_shell_crop, spacing_xyz, iso_spacing)
        shells_iso = portal_shell_iso | hepatic_shell_iso
        union_iso = liver_iso | shells_iso
        write_mask_to_stl(liver_iso, iso_spacing, liver_only_path, smoothing_iterations=args.smoothing_iterations)
        write_mask_to_stl(shells_iso, iso_spacing, shells_only_path, smoothing_iterations=args.smoothing_iterations)
        write_mask_to_stl(union_iso, iso_spacing, merged_path, smoothing_iterations=args.smoothing_iterations)
    else:
        # Native grid, no resample
        shells_native = portal_shell_crop | hepatic_shell_crop
        union_native = liver_crop | shells_native
        # Use native spacing from liver_img
        native_spacing = float(np.linalg.norm(liver_img.affine[:3, 0]))
        write_mask_to_stl(liver_crop, native_spacing, liver_only_path, smoothing_iterations=args.smoothing_iterations)
        write_mask_to_stl(shells_native, native_spacing, shells_only_path, smoothing_iterations=args.smoothing_iterations)
        write_mask_to_stl(union_native, native_spacing, merged_path, smoothing_iterations=args.smoothing_iterations)

    if args.out:
        # also copy/alias merged output to requested path
        import shutil
        shutil.copyfile(merged_path, args.out)
        print(f"[ok] Copied merged output to {args.out}")


if __name__ == "__main__":
    main()
