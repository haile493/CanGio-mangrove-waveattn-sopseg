import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
from src.dataset import discover_scenes, CanGioDataset
from scipy.ndimage import grey_closing, label
from skimage.measure import regionprops

scenes = discover_scenes('data', source='cut')
train_scenes = [s for s in scenes if not any(y in s['year_folder'] for y in ['2011', '2014']) and s.get('groundtruth_path')]
ds = CanGioDataset(train_scenes, include_indices=True, augment=False)

for i in range(len(ds)):
    img, gt, meta = ds[i]
    nbr = img[7].numpy() # [0, 1]
    ndvi = img[6].numpy()
    gt_np = (gt.numpy().squeeze() > 0.5).astype(np.uint8)
    
    canopy_surface = nbr
    filled = grey_closing(canopy_surface, size=(9, 9))
    depr = filled - canopy_surface
    
    # Exclude open water bodies
    forest_mask = (ndvi > 0.25)
    cand_mask = ((depr > 0.12) & forest_mask).astype(np.uint8)
    
    labeled, num = label(cand_mask)
    regions = [r for r in regionprops(labeled) if 3 <= r.area <= 400]
    
    tp = int(np.logical_and(cand_mask == 1, gt_np == 1).sum())
    fp = int(np.logical_and(cand_mask == 1, gt_np == 0).sum())
    fn = int(np.logical_and(cand_mask == 0, gt_np == 1).sum())
    iou = tp / (tp + fp + fn + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    prec = tp / (tp + fp + 1e-7)
    print(f"{meta['scene_id']}: {len(regions)} boxes, GT: {gt_np.sum()}, Cand IoU: {iou:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
