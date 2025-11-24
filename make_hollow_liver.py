import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
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

def marching_cubes_to_stl(vtk_image, iso_value, out_path):
    # For binary data, vtkDiscreteMarchingCubes is often more robust
    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, iso_value)  # label value to extract
    mc.Update()

    # Optional: smooth (can comment out if you prefer raw)
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(mc.GetOutputPort())
    smooth.SetNumberOfIterations(30)
    smooth.SetRelaxationFactor(0.1)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOff()
    smooth.Update()

    # Write STL
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(out_path)
    writer.SetInputConnection(smooth.GetOutputPort())
    writer.Write()
    print(f"[ok] Wrote STL: {out_path}")

def make_hollow_liver(
    liver_path,
    vessels_path,
    out_stl="hollow_liver.stl",
    vessel_dilation_mm=1.0,
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

    # Compute voxel spacing from affine
    affine = liver_img.affine
    spacing_x = np.linalg.norm(affine[0, :3])
    spacing_y = np.linalg.norm(affine[1, :3])
    spacing_z = np.linalg.norm(affine[2, :3])
    spacing = np.array([spacing_z, spacing_y, spacing_x])  # (Z, Y, X) spacing

    # Convert dilation in mm → approx number of voxels
    if vessel_dilation_mm > 0:
        # Rough: use mean voxel size
        mean_voxel = float(np.mean(spacing))
        iters = max(1, int(round(vessel_dilation_mm / mean_voxel)))
        print(f"[info] Dilating vessel mask by ~{vessel_dilation_mm} mm (~{iters} iterations).")
        structure = np.ones((3, 3, 3), dtype=bool)
        vessel_mask = binary_dilation(vessel_mask.astype(bool), structure=structure, iterations=iters).astype(np.uint8)
    else:
        print("[info] No vessel dilation applied.")

    # Boolean subtraction: liver minus vessels
    liver_with_cavity = liver_mask.astype(bool) & (~vessel_mask.astype(bool))
    liver_with_cavity = liver_with_cavity.astype(np.uint8)

    print(f"[info] Liver voxels before: {liver_mask.sum()}, after subtraction: {liver_with_cavity.sum()}")

    # Convert to VTK image for meshing
    vtk_image = numpy_to_vtk_image(liver_with_cavity, affine)

    # Marching cubes: label value is 1
    marching_cubes_to_stl(vtk_image, iso_value=1, out_path=out_stl)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a hollow liver STL by subtracting vessel mask from liver mask."
    )
    parser.add_argument("liver_nii", help="Path to liver mask NIfTI (e.g. liver.nii.gz)")
    parser.add_argument("vessels_nii", help="Path to vessel mask NIfTI (e.g. vessels.nii.gz)")
    parser.add_argument("--out", default="hollow_liver.stl", help="Output STL file name")
    parser.add_argument("--vessel-dilation-mm", type=float, default=1.0,
                        help="Approximate vessel dilation in mm before subtraction (default: 1.0)")
    parser.add_argument("--liver-label", type=int, default=None,
                        help="If liver NIfTI is a label map, specify liver label value")
    parser.add_argument("--vessel-label", type=int, default=None,
                        help="If vessel NIfTI is a label map, specify vessel label value")

    args = parser.parse_args()

    make_hollow_liver(
        liver_path=args.liver_nii,
        vessels_path=args.vessels_nii,
        out_stl=args.out,
        vessel_dilation_mm=args.vessel_dilation_mm,
        liver_label=args.liver_label,
        vessel_label=args.vessel_label,
    )
