"""Per-image independent point clouds from the frozen metric model, plus reference poses.

Key isolation rule: this script writes two separate areas.
  agent_dir/  = only RGB, intrinsics, and the cloud in ITS OWN camera frame.
  eval_dir/   = reference poses, GT, COLMAP cloud. Never read by the agent.
"""
import json, pathlib, numpy as np, cv2, sys
sys.path.insert(0, '.')
from src.config import load_config
from src.backends.da3 import DA3Backend
sys.path.insert(0, 'scripts')
from crossview_scale import read_colmap

ROOT = pathlib.Path('.')
CACHE = ROOT/'cache/hpc/depths'
AGENT = ROOT/'mvrecon/agent_data'
EVAL = ROOT/'mvrecon/eval_data'
for d in (AGENT, EVAL): d.mkdir(parents=True, exist_ok=True)

rows = {json.loads(l)['sample_id']: json.loads(l)
        for l in open('manifests/formal_all.jsonl') if l.strip()}
cfg = load_config('configs/pilot.yaml')

def build_scene(scene, sid_list):
    calib = read_colmap(pathlib.Path(f'D:/..') if False else
                        pathlib.Path(f'E:/depth_calib_data/eth3d_extracted/{scene}/dslr_calibration_jpg'))
    ag = AGENT/scene; ev = EVAL/scene
    ag.mkdir(exist_ok=True); ev.mkdir(exist_ok=True)
    meta_agent = {'scene': scene, 'images': []}
    meta_eval = {'scene': scene, 'images': [], 'reference': 'COLMAP SfM (metric by calibration targets)'}
    for sid in sid_list:
        r = rows[sid]
        npz = CACHE/f'{sid}__B.npz'
        if not npz.exists(): continue
        sd = json.loads((CACHE/f'{sid}__B.json').read_text())
        d = np.load(npz)['depth'].astype(np.float64)
        Hp, Wp = sd['processed_shape']
        K = np.asarray(r['K'], float)
        f = ((K[0,0]*Wp/r['width']) + (K[1,1]*Hp/r['height']))/2/300.0
        # depth at native resolution, in the model's metric convention
        dnat = cv2.resize(d, (r['width'], r['height']))*f
        secs = round(pathlib.Path(r['image']).stem + '', 0) if False else None
        np.save(ag/f'{sid}_depth.npy', dnat.astype(np.float32))
        _ = None
        # copy RGB into the agent area
        src = pathlib.Path(r['dataset_root'])/r['image']
        if src.exists():
            import shutil; shutil.copy(src, ag/f'{sid}.png')
        meta_agent['images'].append({
            'sid': sid, 'K': K.tolist(), 'W': r['width'], 'H': r['height'],
            'depth_units': 'metres (model convention, focal/300 applied)',
            'depth_convention': 'z along optical axis',
            'depth_file': f'{sid}_depth.npy', 'rgb_file': f'{sid}.png'})
        # reference pose: world->cam (R,t) from COLMAP
        src_name = pathlib.Path(r['source_image']).name if r.get('source_image') else None
        if src_name and src_name in calib:
            R, t, Kc, Wc, Hc = calib[src_name]
            meta_eval['images'].append({'sid': sid, 'R': R.tolist(), 't': t.tolist(),
                                        'source_image': src_name})
    (ag/'meta.json').write_text(json.dumps(meta_agent, indent=1))
    (ev/'meta.json').write_text(json.dumps(meta_eval, indent=1))
    return len(meta_agent['images']), len(meta_eval['images'])

# pick 4 views per scene, spread across the sequence (not just adjacent)
import collections
by = collections.defaultdict(list)
for sid, r in rows.items():
    if r['dataset']=='eth3d': by[r['scene_id']].append(sid)
print(f"{'scene':<15}{'n_img':>6}{'n_pose':>7}{'picked':>7}")
for scene in sorted(by):
    allids = sorted(by[scene], key=lambda s: int(s.split('_')[1]))
    # spread: take indices evenly across the full list
    # take up to 10 candidates spread across the sequence; subset selection
    # then picks overlapping ones from these
    k = min(10, len(allids))
    pick = [allids[int(round(i*(len(allids)-1)/(k-1)))] for i in range(k)] if k > 1 else allids
    a,e = build_scene(scene, pick)
    print(f'{scene:<15}{a:>6}{e:>7}{len(pick):>7}')
