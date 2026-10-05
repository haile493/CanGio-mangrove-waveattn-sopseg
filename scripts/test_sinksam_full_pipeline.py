import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from src.dataset import discover_scenes, CanGioDataset
from src.evaluate import compute_binary_metrics
from scipy.ndimage import grey_closing, label
from skimage.measure import regionprops
from segment_anything import sam_model_registry, SamPredictor

def polsby_popper(r):
    if r.perimeter == 0:
        return 1.0
    return (4 * np.pi * r.area) / (r.perimeter ** 2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = sam_model_registry['vit_b'](checkpoint='results/models/sam_vit_b_01ec64.pth').to(device)
predictor = SamPredictor(sam)

scenes = discover_scenes('data', source='cut')
train_scenes = [s for s in scenes if not any(y in s['year_folder'] for y in ['2011', '2014']) and s.get('groundtruth_path')]
ds = CanGioDataset(train_scenes, include_indices=True, augment=False)

for i in range(len(ds)):
    img, gt, meta = ds[i]
    gt_np = (gt.numpy().squeeze() > 0.5).astype(np.uint8)
    
    # 1. Canopy disturbance depression
    nbr = img[7].numpy()
    ndvi = img[6].numpy()
    canopy_surface = nbr
    filled = grey_closing(canopy_surface, size=(9, 9))
    depr = filled - canopy_surface
    
    forest_mask = (ndvi > 0.25)
    cand_mask = ((depr > 0.12) & forest_mask).astype(np.uint8)
    
    labeled, num = label(cand_mask)
    regions = [r for r in regionprops(labeled) if 3 <= r.area <= 400]
    
    # 2. Extract bounding boxes
    boxes = []
    for r in regions:
        y1, x1, y2, x2 = r.bbox
        # Add small margin
        boxes.append([max(0, x1 - 2), max(0, y1 - 2), min(124, x2 + 2), min(124, y2 + 2)])
        
    if not boxes:
        print(f"{meta['scene_id']}: No boxes")
        continue
        
    boxes_tensor = torch.tensor(boxes, device=device, dtype=torch.float32)
    
    # False-color composite for SAM: NIR, Red, Green
    nir = img[3].numpy()
    red = img[2].numpy()
    green = img[1].numpy()
    comp = np.stack([nir, red, green], axis=-1)
    comp = ((comp - comp.min()) / (comp.max() - comp.min() + 1e-7) * 255.0).astype(np.uint8)
    predictor.set_image(comp)
    
    # 3. Monte Carlo CWBJ simulation (N=25)
    N = 25
    stacked_masks = []
    H, W = 124, 124
    for _ in range(N):
        jitter = torch.randint(-3, 4, boxes_tensor.shape, device=device).float()
        j_boxes = boxes_tensor + jitter
        j_boxes[:, 0] = torch.clamp(j_boxes[:, 0], 0, W)
        j_boxes[:, 1] = torch.clamp(j_boxes[:, 1], 0, H)
        j_boxes[:, 2] = torch.clamp(j_boxes[:, 2], 0, W)
        j_boxes[:, 3] = torch.clamp(j_boxes[:, 3], 0, H)
        
        t_boxes = predictor.transform.apply_boxes_torch(j_boxes, (H, W))
        m_out, _, _ = predictor.predict_torch(None, None, boxes=t_boxes, multimask_output=False)
        comb, _ = torch.max(m_out.squeeze(1).float(), dim=0)
        stacked_masks.append(comb.cpu().numpy())
        
    mean_mask = np.mean(stacked_masks, axis=0)
    bin_mask = (mean_mask >= 0.5).astype(np.uint8)
    
    # 4. Polsby-Popper shape filtering
    l_mask, _ = label(bin_mask)
    filtered = np.zeros_like(bin_mask)
    for r in regionprops(l_mask):
        # Keep reasonably compact and non-spurious objects
        if r.area >= 4 and polsby_popper(r) >= 0.15:
            filtered[r.coords[:, 0], r.coords[:, 1]] = 1
            
    # Also intersect with the depression zone so SAM doesn't bleed into healthy canopy
    filtered = filtered * cand_mask
    
    m_raw = compute_binary_metrics(cand_mask.flatten(), gt_np.flatten())
    m_sam = compute_binary_metrics(filtered.flatten(), gt_np.flatten())
    print(f"{meta['scene_id']}: Raw Cand IoU={m_raw['iou']:.4f}, Prec={m_raw['precision']:.4f} | SinkSAM Refined IoU={m_sam['iou']:.4f}, Prec={m_sam['precision']:.4f}, Rec={m_sam['recall']:.4f}, Dice={m_sam['dice']:.4f}")
