"""
使用 VMTK 从气道分割 mask 提取中心线树，并导出 JSON。

运行说明
--------
环境: 需在已安装 VMTK 的 Conda 环境中运行（本机默认 Python 通常无 vmtk）:

  conda create -n vmtk_env -c conda-forge vmtk python=3.9
  conda activate vmtk_env
  pip install nibabel numpy scipy vtk

输入 / 输出路径（与 segmentation pipeline 输出布局一致）:
  读取: {病例根目录}/final/airway/*.nii.gz
  写入: {病例根目录}/final/airway_tree/<与 nii 同名的>.json

单病例模式:

  python tree_vmtk.py --base_dir data/0004data

  若 final/airway 下只有一个 .nii.gz，也可把该目录内全部 nii 批量处理。

批量模式（处理多个病例子文件夹）:

  python tree_vmtk.py --base_dir data --batch

  目录结构示例:
    output/
    ├── CASE052_0000/final/airway/*.nii.gz  ->  .../airway_tree/*.json
    └── CASE053_0000/final/airway/*.nii.gz  ->  .../airway_tree/*.json

  与 run_segmentation_pipeline.py --batch 的 --output_dir 对应；
  无 airway mask 的病例会被跳过，单个病例失败不中断整批。

可视化: 生成 JSON 后可用 visualize_tree_vmtk.py 查看中心线。

查看参数: python tree_vmtk.py -h
（需在已安装 vmtk 的环境中执行 -h，否则会在导入阶段退出）
"""

import argparse
import os
import sys
import json
import math
import time
import numpy as np
from scipy.interpolate import splprep, splev
import vtk
from vtk.util.numpy_support import vtk_to_numpy
# from visualize_aligned import visualize_aligned

try:
    from vmtk import vmtkscripts
except ImportError:
    print("==========================================================")
    print("[ERROR] VMTK (Vascular Modeling Toolkit) is not installed.")
    print("==========================================================")
    print("VMTK provides robust centerline extraction but requires a specific environment.")
    print("Please install VMTK using Conda. Open your Anaconda Prompt and run:")
    print("")
    print("  conda create -n vmtk_env -c conda-forge vmtk python=3.9")
    print("  conda activate vmtk_env")
    print("  pip install nibabel numpy")
    print("")
    print("Then run this script again using the new environment:")
    print("  python tree_vmtk.py --base_dir <病例输出根目录>")
    print("  python tree_vmtk.py --base_dir <输出根目录> --batch")
    print("==========================================================")
    sys.exit(1)

def get_trachea_inlet(nii_path):
    """
    Fast VTK-based method to find the trachea inlet and bounding box.
    Replaces slow scipy.ndimage volume processing.
    """
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path)
    reader.Update()
    image_data = reader.GetOutput()
    
    # Extract surface
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(image_data)
    marchingCubes.SetValue(0, 0.5)
    marchingCubes.Update()
    
    # Find largest connected component on the surface directly
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(marchingCubes.GetOutput())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    largest_pd = conn.GetOutput()
    
    if largest_pd.GetNumberOfPoints() == 0:
        raise ValueError("Mask is completely empty!")
        
    # Get physical bounding box
    bounds = largest_pd.GetBounds()
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    
    def get_voxel_range(b_min, b_max, orig, spac):
        idx1 = (b_min - orig) / spac
        idx2 = (b_max - orig) / spac
        return int(math.ceil(min(idx1, idx2))), int(math.floor(max(idx1, idx2)))
        
    i_min, i_max = get_voxel_range(bounds[0], bounds[1], origin[0], spacing[0])
    j_min, j_max = get_voxel_range(bounds[2], bounds[3], origin[1], spacing[1])
    k_min, k_max = get_voxel_range(bounds[4], bounds[5], origin[2], spacing[2])
    
    bbox = (i_min, i_max, j_min, j_max, k_min, k_max)
    
    # Find highest Z point
    points = vtk_to_numpy(largest_pd.GetPoints().GetData())
    z_max_idx = np.argmax(points[:, 2])
    inlet_pt = points[z_max_idx].tolist()
    
    return inlet_pt, bbox, image_data

def prune_fake_roots(cell_info):
    """
    Post-process the extracted tree to remove fake root segments caused by noisy segmentations.
    A fake root is typically a tiny bump at the top of the trachea with a very small radius.
    We identify fake roots by their small radius.
    We iteratively remove them and move the root down to the child with the largest subtree.
    """
    tree_dict = {item['index']: item for item in cell_info}
    
    children_map = {idx: [] for idx in tree_dict}
    root_idx = None
    for item in cell_info:
        idx = item['index']
        f_idx = item['fatherindex']
        if f_idx == "-1":
            root_idx = idx
        elif f_idx in children_map:
            children_map[f_idx].append(idx)
            
    if root_idx is None:
        return cell_info
        
    def count_descendants(node_idx):
        count = 0
        for child in children_map.get(node_idx, []):
            count += 1 + count_descendants(child)
        return count

    def get_segment_length(member):
        if len(member) < 2: return 0.0
        dist = 0.0
        for i in range(1, len(member)):
            p1, p2 = member[i-1], member[i]
            dist += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
        return dist

    pruned_count = 0
    while root_idx is not None:
        root_item = tree_dict[root_idx]
        
        radii = [pt[3] for pt in root_item['member'] if len(pt) > 3]
        avg_radius = sum(radii) / len(radii) if radii else 999.0
        length = get_segment_length(root_item['member'])
        
        # Criteria for a fake root:
        # 1. Radius is too small for a trachea (< 3.0)
        # 2. Or radius is marginal (< 4.0) but length is extremely short (< 10.0)
        is_fake = (avg_radius < 3.0) or (avg_radius < 4.0 and length < 10.0)
        
        if is_fake:
            children = children_map.get(root_idx, [])
            if not children:
                break # Cannot prune if it's the only segment
                
            # Pick the child with the largest subtree as the new root
            best_child = max(children, key=lambda c: count_descendants(c))
            tree_dict[best_child]['fatherindex'] = "-1"
            
            def delete_subtree(n_idx):
                if n_idx in tree_dict:
                    del tree_dict[n_idx]
                for c in children_map.get(n_idx, []):
                    delete_subtree(c)
                    
            del tree_dict[root_idx]
            for child in children:
                if child != best_child:
                    delete_subtree(child)
                    
            root_idx = best_child
            pruned_count += 1
            print(f"  [Prune] Removed fake root (length: {length:.1f}, avg_radius: {avg_radius:.2f}). New root: {best_child}")
        else:
            print(f"  [Prune] Found true root {root_idx} (length: {length:.1f}, avg_radius: {avg_radius:.2f})")
            break

    if pruned_count == 0:
        return cell_info
        
    print(f"  [Prune] Total {pruned_count} fake root(s) removed.")
    
    # Re-index to ensure indices are continuous from 0
    new_children_map = {idx: [] for idx in tree_dict}
    for item in tree_dict.values():
        f_idx = item['fatherindex']
        if f_idx != "-1" and f_idx in new_children_map:
            new_children_map[f_idx].append(item['index'])
            
    new_cell_info = []
    old_to_new = {}
    queue = [(root_idx, "-1")]
    new_idx_counter = 0
    
    while queue:
        curr_old_idx, new_father_idx = queue.pop(0)
        
        old_item = tree_dict[curr_old_idx]
        new_idx_str = str(new_idx_counter)
        old_to_new[curr_old_idx] = new_idx_str
        
        new_item = {
            "index": new_idx_str,
            "start": old_item['start'],
            "end": old_item['end'],
            "member": old_item['member'],
            "fatherindex": new_father_idx
        }
        new_cell_info.append(new_item)
        new_idx_counter += 1
        
        for child in new_children_map.get(curr_old_idx, []):
            queue.append((child, new_idx_str))
            
    return new_cell_info

def interpolate_segment(member, target_spacing=1.0, include_radius=False):
    """
    Interpolate a single segment's member points (x, y, z, r) using cubic splines.
    Points will be resampled such that the distance between consecutive points
    is approximately `target_spacing`.
    """
    if len(member) < 2:
        return member
        
    pts = np.array(member)
    
    # Calculate cumulative distance along the segment
    diffs = np.diff(pts[:, :3], axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_dists[-1]
    
    if total_length == 0:
        return member
        
    # We need at least 4 points for a cubic spline (k=3). 
    # If we have 2 or 3 points, we drop to linear (k=1) or quadratic (k=2).
    k = min(3, len(pts) - 1)
    
    # We still use spline for X, Y, Z coordinates to keep the curve smooth
    s_val = len(pts) * 0.1
    tck_xyz, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], u=cum_dists, k=k, s=s_val)
    
    # Create new parameter array based on target spacing
    num_new_pts = max(2, int(np.ceil(total_length / target_spacing)) + 1)
    u_new = np.linspace(0, total_length, num_new_pts)
    
    # Evaluate XYZ spline
    new_x, new_y, new_z = splev(u_new, tck_xyz)
    
    new_member = []
    
    if include_radius and pts.shape[1] > 3:
        # --- Radius Smoothing Logic ---
        # Only use the valid start and end radii of this segment and linearly interpolate the radius 
        # based on the cumulative distance. This completely removes radius jitter/fluctuations.
        start_r = pts[0, 3]
        end_r = pts[-1, 3]
        
        # Validation: In a normal vascular/airway tree, the radius should generally decrease 
        # or stay roughly the same from parent to child (start to end of a segment).
        if end_r > start_r:
            if end_r > start_r * 1.05:
                end_r = start_r
                
        # Linearly interpolate the radius from start to end based on distance
        new_r = np.interp(u_new, [0, total_length], [start_r, end_r])
        
        # Reconstruct the interpolated member list with radius
        for i in range(num_new_pts):
            new_member.append([
                round(float(new_x[i]), 2),
                round(float(new_y[i]), 2),
                round(float(new_z[i]), 2),
                round(float(new_r[i]), 2)
            ])
    else:
        # Reconstruct the interpolated member list without radius
        for i in range(num_new_pts):
            new_member.append([
                round(float(new_x[i]), 2),
                round(float(new_y[i]), 2),
                round(float(new_z[i]), 2)
            ])
        
    return new_member

def extract_airway_tree_vmtk(segmentation_path, output_json_path, include_radius=False):
    total_t0 = time.time()
    print(f"Loading mask from {segmentation_path}...")
    
    t0 = time.time()
    inlet_pt, bbox, image_data = get_trachea_inlet(segmentation_path)
    t1 = time.time()
    print(f"Detected Trachea inlet physical coordinate: {inlet_pt} (Time: {t1-t0:.2f}s)")

    # 1.5 Crop to the exact bounding box of the airway
    t0 = time.time()
    print(f"Cropping volume to bounding box {bbox} to force open boundaries...")
    voi = vtk.vtkExtractVOI()
    voi.SetInputData(image_data)
    voi.SetVOI(*bbox)
    voi.Update()
    t1 = time.time()
    print(f"VTK Image Cropping done (Time: {t1-t0:.2f}s)")

    # 2. Extract surface using Marching Cubes
    t0 = time.time()
    print("Extracting surface using Marching Cubes...")
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputConnection(voi.GetOutputPort())
    marchingCubes.SetValue(0, 0.5)
    marchingCubes.Update()
    surface = marchingCubes.GetOutput()
    t1 = time.time()
    print(f"Marching Cubes done (Time: {t1-t0:.2f}s)")

    # 3. Clean surface
    t0 = time.time()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()
    surface = cleaner.GetOutput()

    # Optional: smooth surface to avoid centerline branches going into rough bumps
    print("Smoothing surface...")
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(20)
    smoother.SetPassBand(0.1)
    smoother.Update()
    surface = smoother.GetOutput()
    t1 = time.time()
    print(f"Surface Cleaning & Smoothing done (Time: {t1-t0:.2f}s)")

    # 4. Extract topological skeleton using VMTK Network Extraction
    t0 = time.time()
    print("Extracting skeleton network using vmtkNetworkExtraction (fast and robust)...")
    network = vmtkscripts.vmtkNetworkExtraction()
    network.Surface = surface
    network.Execute()
    cl_polydata = network.Network
    t1 = time.time()
    print(f"VMTK Network Extraction done (Time: {t1-t0:.2f}s)")
    
    # 5. Parse the network graph into a tree structure based on spatial connectivity
    t0 = time.time()
    print("Building tree topology from the network graph...")
    num_cells = cl_polydata.GetNumberOfCells()
    points = cl_polydata.GetPoints()
    
    # Try to get the radius array from point data
    radius_array = cl_polydata.GetPointData().GetArray("Radius")
    if radius_array is None:
        radius_array = cl_polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    
    raw_segments = []
    for i in range(num_cells):
        cell = cl_polydata.GetCell(i)
        n_pts = cell.GetNumberOfPoints()
        pts = [cell.GetPointId(j) for j in range(n_pts)]
        raw_segments.append(pts)
        
    if not raw_segments:
        raise ValueError("Failed to find a valid root segment: VMTK Network Extraction produced 0 segments. The input mask might be fragmented (multi-label gaps) or touch the image boundaries (open surface).")
        
    # Find root segment (closest to trachea inlet)
    min_dist = float('inf')
    root_seg_idx = -1
    root_pt_idx_in_seg = -1
    
    for i, seg in enumerate(raw_segments):
        for j, pid in enumerate(seg):
            coord = points.GetPoint(pid)
            dist = math.sqrt(sum((coord[k] - inlet_pt[k])**2 for k in range(3)))
            if dist < min_dist:
                min_dist = dist
                root_seg_idx = i
                root_pt_idx_in_seg = j
                
    if root_seg_idx == -1:
        raise ValueError("Failed to find a valid root segment.")
        
    # Map endpoints to segments for fast traversal using coordinates to handle duplicate points
    pt_to_segs = {}
    
    def get_coord_key(pid):
        coord = points.GetPoint(pid)
        # Rounding to 2 decimal places (10 microns) avoids float precision issues
        # where bifurcations might have slightly different coordinates
        return (round(coord[0], 2), round(coord[1], 2), round(coord[2], 2))

    for i, seg in enumerate(raw_segments):
        for pid in [seg[0], seg[-1]]:
            coord_key = get_coord_key(pid)
            pt_to_segs.setdefault(coord_key, []).append(i)
            
    oriented_segments = {}
    visited_segs = set()
    
    # Determine the starting point of the root segment
    root_start_pid = raw_segments[root_seg_idx][0]
    if root_pt_idx_in_seg > len(raw_segments[root_seg_idx]) / 2:
        root_start_pid = raw_segments[root_seg_idx][-1]
        
    # BFS to traverse the tree and assign father indices
    # queue stores tuples of: (current_seg_idx, father_seg_idx, connection_coord_key)
    queue = [(root_seg_idx, -1, get_coord_key(root_start_pid))]
    
    new_idx = 0
    seg_idx_mapping = {}  # Maps original cell index to new contiguous index
    
    while queue:
        curr_idx, father_idx, conn_key = queue.pop(0)
        if curr_idx in visited_segs:
            continue
        visited_segs.add(curr_idx)
        
        seg_pts = raw_segments[curr_idx]
        
        # Orient segment away from the connection point
        if get_coord_key(seg_pts[-1]) == conn_key:
            seg_pts.reverse()
            
        seg_idx_mapping[curr_idx] = new_idx
        oriented_segments[new_idx] = {
            "pts": seg_pts,
            "father": father_idx  # This is already mapped to the new index when appended to queue
        }
        new_idx += 1
        
        end_key = get_coord_key(seg_pts[-1])
        children = [idx for idx in pt_to_segs.get(end_key, []) if idx not in visited_segs]
        
        # STRICT BINARY TREE ENFORCEMENT
        # If a node has more than 2 branches (trifurcation or more), we must artificially
        # split it to maintain the binary tree property expected by visualize_tree.py
        current_father_idx = seg_idx_mapping[curr_idx]
        
        while len(children) > 2:
            # Take the FIRST child to be a direct child of the current father
            c1 = children.pop(0)
            queue.append((c1, current_father_idx, end_key))
            
            # Create a zero-length dummy segment for the remaining children
            # This turns a trifurcation into two consecutive bifurcations
            dummy_idx = new_idx
            new_idx += 1
            
            dummy_pid = seg_pts[-1]
            oriented_segments[dummy_idx] = {
                "pts": [dummy_pid, dummy_pid], # Zero length
                "father": current_father_idx
            }
            
            # The remaining children will now branch off the dummy segment
            current_father_idx = dummy_idx
            
        # Normal binary or unary branching
        for child_idx in children:
            queue.append((child_idx, current_father_idx, end_key))
            
    # 6. Convert to JSON format with voxel coordinates
    print("Formatting tree into JSON...")
    cell_info = []
    
    # We don't need affine inverse anymore.
    # vtkNIFTIImageReader reads the image and sets the Origin and Spacing on the vtkImageData.
    # When Marching Cubes and vmtkNetworkExtraction run, they produce coordinates 
    # that are STILL in the VTK world space (which depends on Origin and Spacing).
    # To get back to pure integer image indices (IJK), we need to use the image's Origin and Spacing.
    
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    
    total_original_points = 0
    total_filtered_points = 0
    
    for idx, data in oriented_segments.items():
        member = []
        last_xyz = None
        
        for pid in data["pts"]:
            coord = points.GetPoint(pid)
            
            # Convert VTK world coordinate back to image index:
            # index = (world - origin) / spacing
            i = (coord[0] - origin[0]) / spacing[0]
            j = (coord[1] - origin[1]) / spacing[1]
            k = (coord[2] - origin[2]) / spacing[2]
            
            # Keep floating-point precision to avoid stair-step artifacts
            # which cause wild oscillations during spline interpolation
            current_xyz = (round(i, 3), round(j, 3), round(k, 3))
            
            total_original_points += 1
            
            # Deduplicate: only add if XYZ is different from the last point
            if current_xyz != last_xyz:
                xyz_list = list(current_xyz)
                if radius_array:
                    # Add the radius as the 4th element (keep 2 decimal places for neatness)
                    radius_val = radius_array.GetValue(pid)
                    radius_val_rounded = round(radius_val, 2)
                    
                    # Filter out points with zero radius (boundary artifacts from VMTK)
                    if radius_val_rounded <= 0.0:
                        continue
                        
                    xyz_list.append(radius_val_rounded)
                    
                member.append(xyz_list)
                last_xyz = current_xyz
                total_filtered_points += 1
            
        if not member:
            continue
            
        cell_info.append({
            "index": str(idx),
            "start": member[0],
            "end": member[-1],
            "member": member,
            "fatherindex": str(data["father"])
        })
        
    t1 = time.time()
    print(f"Tree Topology Building, JSON formatting & deduplication done (Time: {t1-t0:.2f}s)")
    print(f"  -> Removed {total_original_points - total_filtered_points} duplicate points.")

    # --- ADD PRUNING HERE ---
    print("\nPruning fake root segments...")
    cell_info = prune_fake_roots(cell_info)
    # ------------------------

    # --- ADD INTERPOLATION HERE ---
    print("\nInterpolating sparse segments using cubic splines...")
    target_spacing = 2.0  # Increased spacing to make it less dense
    for item in cell_info:
        if len(item['member']) > 1:
            interpolated_member = interpolate_segment(item['member'], target_spacing, include_radius)
            item['member'] = interpolated_member
            # Update start and end points to match the new interpolated boundaries
            item['start'] = interpolated_member[0]
            item['end'] = interpolated_member[-1]
    # ------------------------------

    # Save to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cell_info, f, indent=4)
        
    total_t1 = time.time()
    print(f"\n==========================================================")
    print(f"SUCCESS: Extracted {len(cell_info)} segments.")
    print(f"Total Execution Time: {total_t1-total_t0:.2f}s")
    print(f"Saved JSON to: {output_json_path}")
    print(f"==========================================================")

def list_batch_case_dirs(batch_root: str) -> list[str]:
    """列出批量根目录下的病例子文件夹（已排序）。"""
    batch_root = os.path.abspath(batch_root.rstrip(os.sep))
    if not os.path.isdir(batch_root):
        raise NotADirectoryError(f"批量目录不存在: {batch_root}")

    case_dirs: list[str] = []
    for name in sorted(os.listdir(batch_root)):
        if name.startswith("."):
            continue
        path = os.path.join(batch_root, name)
        if os.path.isdir(path):
            case_dirs.append(path)

    if not case_dirs:
        raise ValueError(f"未在 {batch_root} 下找到病例子文件夹")
    return case_dirs


def collect_airway_nii_files(input_path: str) -> list[str]:
    """收集单个病例下待处理的 .nii.gz（文件或目录）。"""
    if os.path.isfile(input_path) and input_path.endswith(".nii.gz"):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".nii.gz")
        )
        return files
    return []


def run_single_case(case_root: str, *, export_radius: bool = False) -> tuple[int, int]:
    """
    处理单个病例目录。
    返回 (成功文件数, 待处理文件总数)。
    """
    case_root = os.path.abspath(case_root)
    case_name = os.path.basename(case_root.rstrip(os.sep))
    input_path = os.path.join(case_root, "final", "airway")
    output_dir = os.path.join(case_root, "final", "airway_tree")

    input_files = collect_airway_nii_files(input_path)
    if not input_files:
        print(f"[跳过] {case_name}: 未在 {input_path} 找到 .nii.gz")
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    print(
        f"病例 {case_name}: 在 {input_path} 找到 {len(input_files)} 个文件，"
        f"输出 -> {output_dir}"
    )

    success = 0
    for full_path in input_files:
        filename = os.path.basename(full_path)
        json_filename = filename.replace(".nii.gz", ".json")
        output_json = os.path.join(output_dir, json_filename)

        print(f"\n==========================================================")
        print(f"Processing: {case_name} / {filename}")
        print(f"==========================================================")

        extract_airway_tree_vmtk(full_path, output_json, include_radius=export_radius)
        success += 1

    return success, len(input_files)


def run_batch(batch_root: str, *, export_radius: bool = False) -> int:
    """批量处理 batch_root 下每个病例子文件夹。"""
    case_dirs = list_batch_case_dirs(batch_root)
    total_cases = len(case_dirs)
    completed_cases = 0
    total_files = 0
    success_files = 0
    failures: list[tuple[str, BaseException]] = []

    print(f"批量模式: {batch_root}（共 {total_cases} 个病例）")

    for index, case_dir in enumerate(case_dirs, start=1):
        case_name = os.path.basename(case_dir.rstrip(os.sep))
        print(f"\n{'=' * 60}")
        print(f"[批次 {index}/{total_cases}] {case_name}")
        print(f"{'=' * 60}")
        try:
            ok, n = run_single_case(case_dir, export_radius=export_radius)
            total_files += n
            success_files += ok
            if n > 0 and ok == n:
                completed_cases += 1
        except Exception as exc:
            failures.append((case_name, exc))
            print(f"[失败] {case_name}: {exc}", file=sys.stderr)

    print(f"\n{'=' * 60}")
    print(
        f"批量完成: 病例 {completed_cases}/{total_cases} 全部成功, "
        f"文件 {success_files}/{total_files}"
    )
    if failures:
        print(f"失败病例 {len(failures)}/{total_cases}:", file=sys.stderr)
        for case_name, exc in failures:
            print(f"  ✗ {case_name}: {exc}", file=sys.stderr)
        return 1
    if success_files == 0:
        print("[ERROR] 未处理任何 .nii.gz 文件", file=sys.stderr)
        return 1
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 VMTK 从气道分割 mask 提取中心线树并导出 JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help=(
            "输出目录：单病例时为 {base_dir}/final/airway；"
            "配合 --batch 时为含多个病例子文件夹的根目录（如 pipeline 的 --output_dir）"
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量模式：处理 base_dir 下每个子文件夹为一个病例",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)
    EXPORT_RADIUS = False

    if args.batch:
        raise SystemExit(run_batch(base_dir, export_radius=EXPORT_RADIUS))

    ok, n = run_single_case(base_dir, export_radius=EXPORT_RADIUS)
    if n == 0:
        print(f"[ERROR] 未在 {os.path.join(base_dir, 'final', 'airway')} 找到可处理的 .nii.gz")
        raise SystemExit(1)
    raise SystemExit(0 if ok == n else 1)
