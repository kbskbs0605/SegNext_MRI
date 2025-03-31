import torch
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import zoom, center_of_mass, label

# Function to load NIfTI file using nibabel
def load_nifti(file_path, is_mask=False):
    nii_img = nib.load(file_path)  # Load NIfTI file
    dtype = np.float32  # Load everything as float32 (important for affine_transform)
    
    voxel_data = np.asanyarray(nii_img.dataobj, dtype=dtype)  # Load data as float32
    affine_matrix = nii_img.affine  # Get affine transformation matrix
    voxel_size = nii_img.header.get_zooms()  # Get voxel resolution

    return voxel_data, voxel_size, affine_matrix

def resampling(voxel_data, original_spacing, is_mask=False):
    spacing_array = np.array(original_spacing)

    if np.any(np.isclose(spacing_array, 0.5, atol=1e-1)):
        zoom_factors = [
            2.0 if np.isclose(s, 1.0, atol=0.2) else 1.0 
            for s in spacing_array
        ]

        resampled_data = zoom(
            voxel_data, 
            zoom_factors, 
            order=0 if is_mask else 3
        )

        new_spacing = spacing_array / zoom_factors
        return resampled_data, tuple(new_spacing)
    else:
        return voxel_data, tuple(spacing_array)



# Function to process and save subject data
def process_subject_data(index, img_path, gt_path, pred_path, output_dir, log_file="./data/3D_pt/transformation_log.txt"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load voxel data and metadata
    img, original_spacing_img, affine_img = load_nifti(img_path)
    gt, original_spacing_gt, affine_gt = load_nifti(gt_path, is_mask=True)
    pred, original_spacing_pred, affine_gt = load_nifti(pred_path, is_mask=True)

    # Resample to target spacing
    img_resampled, resulting_spacing = resampling(img, original_spacing_img)
    gt_resampled , resulting_spacing= resampling(gt, original_spacing_gt, is_mask=True)
    pred_resampled , resulting_spacing= resampling(pred, original_spacing_pred, is_mask=True)
    
    gt_resampled = np.round(gt_resampled).astype(np.uint8)
    pred_resampled = np.round(pred_resampled).astype(np.uint8)
    
    # Normalize image intensities (Min-Max scaling)
    img_resampled = (img_resampled - np.min(img_resampled)) / (np.max(img_resampled) + 1e-5)
    
    # Convert to PyTorch tensors
    tensor_img = torch.tensor(img_resampled, dtype=torch.float32)
    tensor_gt = torch.tensor(gt_resampled, dtype=torch.uint8)
    tensor_pred = torch.tensor(pred_resampled, dtype=torch.uint8)
    
    # Save as .pt files
    torch.save(tensor_img, os.path.join(output_dir, f"s{index}_img.pt"))
    torch.save(tensor_gt, os.path.join(output_dir, f"s{index}_gt.pt"))
    torch.save(tensor_pred, os.path.join(output_dir, f"s{index}_pred.pt"))
    

    print(f'shape img: {tensor_img.shape}')
    print(f'shape gt: {tensor_gt.shape}')
    print(f'shape pred: {tensor_pred.shape}')

        # Compute statistics before and after transformation
    img_min, img_max, img_avg = np.min(img), np.max(img), np.mean(img)
    img_resampled_min, img_resampled_max, img_resampled_avg = np.min(img_resampled), np.max(img_resampled), np.mean(img_resampled)
    gt_count = np.sum(gt == 1)
    # gt_affine_count = np.sum(gt_affine == 1)
    gt_resampled_count = np.sum(gt_resampled == 1)

    spheres_str = ""

    mask = gt_resampled == 1
    if np.any(mask):
        # Identify disconnected spheres
        labeled_array, num_spheres = label(mask)

        spheres_str += f"\nNumber of Spheres: {num_spheres}\n"

        for sphere_id in range(1, num_spheres + 1):
            sphere_mask = labeled_array == sphere_id
            num_ones = np.sum(sphere_mask)
            com = center_of_mass(sphere_mask)

            # Estimate diameter assuming a spherical shape
            radius = ((3 * num_ones) / (4 * np.pi)) ** (1/3)
            diameter = 2 * radius

            spheres_str += (
                f"  Sphere {sphere_id}:\n"
                f"    - Center of Mass: {com}\n"
                f"    - Number of `1`s (Voxel Count): {num_ones}\n"
                f"    - Estimated Diameter (in pixels): {diameter:.2f}\n"
            )
    else:
        spheres_str += "\nNo `1`s found in the mask.\n"
    
    # Write transformation details to log file
    with open(log_file, "a") as log:
        log.write(f"Subject {index}\n\n")
        log.write(f"Image: Shape {img.shape} -> {img_resampled.shape}\n")
        log.write(f"Resolution {original_spacing_img} -> {resulting_spacing}\n")
        log.write(f"Min {img_min} -> {img_resampled_min}\n")
        log.write(f"Max {img_max} -> {img_resampled_max}\n")
        log.write(f"Avg {img_avg} -> {img_resampled_avg}\n\n")
        log.write(f"GT Mask: Shape {gt.shape} -> {gt_resampled.shape}\n")
        log.write(f"Resolution {original_spacing_gt} -> {resulting_spacing}\n")
        #log.write(f"Ones Count {gt_count} -> {gt_affine_count} -> {gt_resampled_count}\n")
        log.write(f"Ones Count {gt_count} -> {gt_resampled_count}\n\n")
        log.write(spheres_str)
        log.write("------------------------------\n")
    
    print(f"Saved: s{index}_img.pt & s{index}_gt.pt")


################################################################################################
################################ Example Directory Setting######################################
################################################################################################

input_dir = "/data/human/CMC/sample2"  # Modify with actual path
output_dir = "./data/3D_pt"
os.makedirs(output_dir, exist_ok=True)

log_file = "./data/3D_pt/transformation_log.txt"
if os.path.exists(log_file):
    os.remove(log_file)  # Delete existing log file to start fresh

for i in range(10):
    if i == 7:
        continue
    img_path = os.path.join(input_dir, f"imagesTr/HyperArcS_00{i}_0000.nii.gz")
    gt_path = os.path.join(input_dir, f"labelGroundTruth/HyperArcS_00{i}.nii.gz")
    pred_path = os.path.join(input_dir, f"labelsPredicted/HyperArcS_00{i}.nii.gz")
    process_subject_data(i, img_path, gt_path, pred_path, output_dir)
