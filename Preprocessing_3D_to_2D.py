import torch
import nibabel as nib
import numpy as np
import torch
import os
import cv2  # Using OpenCV for faster image saving

# Output directory
output_dir_img_pt = "./data/image"
output_dir_gt_pt = "./data/GT"

os.makedirs(output_dir_img_pt, exist_ok=True)
os.makedirs(output_dir_gt_pt, exist_ok=True)

output_dir_png = "./data/2D_png"
os.makedirs(output_dir_png, exist_ok=True)

output_dir_raw_png = "./data/2D_raw_png"
os.makedirs(output_dir_raw_png, exist_ok=True)

output_dir_predicted_png = "./data/2D_predicted_mask"
os.makedirs(output_dir_predicted_png, exist_ok=True)


for i in range(10):
    if i == 7 or i == 2:
        continue
    # Load the NIfTI files
    voxelGT = torch.load(f'./data/3D_pt/s{i}_gt.pt')
    voxelImg = torch.load(f'./data/3D_pt/s{i}_img.pt')
    voxelPred = torch.load(f'./data/3D_pt/s{i}_pred.pt')

    # Convert voxelGT to boolean mask (1 -> mask, 0 -> background)
    voxelGT = voxelGT.numpy().astype(bool)
    voxelPred = voxelPred.numpy().astype(bool)
    voxelImg = voxelImg.numpy()

    # def save_slices(voxel_img, voxel_gt, axis, axis_name):
    #     """ Extracts slices where voxelGT has 1s and saves them as .pt and .png files """
    #     indices = np.where(voxel_gt == 1)  # Get all indices where GT is 1
    #     unique_slices = np.unique(indices[axis])  # Get unique slice indices

    #     for idx in unique_slices:
    #         if axis == 0:  # Sagittal (YZ)
    #             img_slice = voxel_img[idx, :, :]
    #             gt_slice = voxel_gt[idx, :, :]
    #         elif axis == 1:  # Coronal (XZ)
    #             img_slice = voxel_img[:, idx, :]
    #             gt_slice = voxel_gt[:, idx, :]
    #         elif axis == 2:  # Axial (XY)
    #             img_slice = voxel_img[:, :, idx]
    #             gt_slice = voxel_gt[:, :, idx]

    #         # Convert to PyTorch tensor
    #         img_tensor = torch.tensor(img_slice, dtype=torch.float32)
    #         gt_tensor = torch.tensor(gt_slice, dtype=torch.uint8)

    #         # Save .pt file
    #         save_path = os.path.join(output_dir_img_pt, f"sub{i}_{axis_name}_{idx:03d}.pt")
    #         torch.save(img_tensor, save_path)

    #         # Save .pt file
    #         save_path = os.path.join(output_dir_gt_pt, f"sub{i}_{axis_name}_{idx:03d}.pt")
    #         torch.save(gt_tensor, save_path)

    #         # Save as PNG file
    #         png_path = os.path.join(output_dir_png, f"sub{i}_{axis_name}_{idx:03d}.png")
    #         save_as_png(img_tensor.numpy(), gt_tensor.numpy(), png_path)

    #         # Save as raw PNG file
    #         raw_png_path = os.path.join(output_dir_raw_png, f"sub{i}_{axis_name}_{idx:03d}.png")
    #         save_raw_png(img_tensor.numpy(), raw_png_path)

    def save_slices(voxel_img, voxel_gt, voxel_predicted, axis, axis_name, min_gt_pixels=20):
        """ 
        Extracts slices where voxelGT has at least `min_gt_pixels` as 1s
        and saves them as .pt and .png files.
        """
        indices = np.where(voxel_gt == 1)  # Get all indices where GT is 1
        unique_slices = np.unique(indices[axis])  # Get unique slice indices

        for idx in unique_slices:
            if axis == 0:  # Sagittal (YZ)
                img_slice = voxel_img[idx, :, :]
                gt_slice = voxel_gt[idx, :, :]
                pred_slice = voxel_predicted[idx, :, :]
            elif axis == 1:  # Coronal (XZ)
                img_slice = voxel_img[:, idx, :]
                gt_slice = voxel_gt[:, idx, :]
                pred_slice = voxel_predicted[:, idx, :]
            elif axis == 2:  # Axial (XY)
                img_slice = voxel_img[:, :, idx]
                gt_slice = voxel_gt[:, :, idx]
                pred_slice = voxel_predicted[:, :, idx]

            # **Check if slice contains at least `min_gt_pixels` GT pixels**
            if np.sum(gt_slice) < min_gt_pixels:
                continue  # Skip this slice if it doesn't meet the requirement

            # Convert to PyTorch tensor
            img_tensor = torch.tensor(img_slice, dtype=torch.float32)
            gt_tensor = torch.tensor(gt_slice, dtype=torch.uint8)
            pred_tensor = torch.tensor(pred_slice, dtype=torch.uint8)

            # Save .pt file
            save_path = os.path.join(output_dir_img_pt, f"sub{i}_{axis_name}_{idx:03d}.pt")
            torch.save(img_tensor, save_path)

            # Save .pt file for GT
            save_path = os.path.join(output_dir_gt_pt, f"sub{i}_{axis_name}_{idx:03d}.pt")
            torch.save(gt_tensor, save_path)

            # Save as PNG file
            png_path = os.path.join(output_dir_png, f"sub{i}_{axis_name}_{idx:03d}.png")
            save_as_png(img_tensor.numpy(), gt_tensor.numpy(), png_path)

            # Save as PNG file
            png_pred_path = os.path.join(output_dir_predicted_png, f"sub{i}_{axis_name}_{idx:03d}.png")
            save_mask_png(img_tensor.numpy(), pred_tensor.numpy(), png_pred_path)

            # Save as raw PNG file
            raw_png_path = os.path.join(output_dir_raw_png, f"sub{i}_{axis_name}_{idx:03d}.png")
            save_raw_png(img_tensor.numpy(), raw_png_path)

    def save_as_png(image, mask, save_path):
        """ Save the image in grayscale with the mask overlaid in 50% green using OpenCV """

        # Normalize the image to [0,255] for visualization
        img_normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255
        img_gray = img_normalized.astype(np.uint8)

        # Convert grayscale to BGR for OpenCV (required for color overlay)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # Create a green mask overlay
        mask_colored = np.zeros_like(img_bgr)
        mask_colored[:, :, 1] = mask * 255  # Green channel

        # Blend the mask with the image using 50% opacity
        blended = cv2.addWeighted(img_bgr, 1.0, mask_colored, 0.3, 0)

        # Save using OpenCV
        cv2.imwrite(save_path, blended)

    def save_mask_png(image, mask, save_path):
        """ Save the image in grayscale with the mask overlaid in 50% green using OpenCV """

        # Normalize the image to [0,255] for visualization
        img_gray = (image * 0).astype(np.uint8)

        # Convert grayscale to BGR for OpenCV (required for color overlay)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # Create a green mask overlay
        mask_colored = np.zeros_like(img_bgr)
        mask_colored[:, :, 1] = mask * 255


        # Save using OpenCV
        cv2.imwrite(save_path, mask_colored)

    def save_raw_png(image, save_path):
        """ Save the image in grayscale using OpenCV without any overlay """

        # Normalize the image to [0,255] for visualization
        img_normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255
        img_gray = img_normalized.astype(np.uint8)

        # Save using OpenCV (grayscale image)
        cv2.imwrite(save_path, img_gray)

    # Process and save slices in three orientations
    save_slices(voxelImg, voxelGT, voxelPred, axis=0, axis_name="YZ")  # Sagittal
    save_slices(voxelImg, voxelGT, voxelPred, axis=1, axis_name="XZ")  # Coronal
    save_slices(voxelImg, voxelGT, voxelPred, axis=2, axis_name="XY")  # Axial

    print(f"Saved 2D slices (pt/png) for subject{i}")
