# NavVis Localization Assets

**Download:** https://www.swisstransfer.com/d/4bc771c8-d56b-498b-8073-7eab54cecf9b

## What’s inside
- Place the downloaded `output` folder into `/mesh_pipeline/`. It already contains the localization map, features, and everything needed to run localization (no need to rebuild).
- New files:
  - `notebooks/hloc_navvis.ipynb`: HLoc pipeline for NavVis with visualizations.
  - `scripts/run_hloc_navvis.py`: Headless HLoc pipeline (use this for integration).

## How to run
- Put test images under `data/test_images/` (e.g., `data/test_images/test1.jpeg`; any HG images work).
- Run headless localization:  
  `python scripts/run_hloc_navvis.py --save-query-plot`  
  The `--save-query-plot` flag writes a query-only 3D pose visualization to `outputs/hloc/<session>/localization/query_poses.html`.
- Or open the notebook for interactive visualizations.

## Notes
- Original HLoc demo: https://colab.research.google.com/drive/1Eqoz-uLTCGeEWtH95FZyVs2vI-qkTOWr#scrollTo=CeL5x1beYrtX (heavily adapted for our larger dataset).
- NavVis tweak: images 280–382 (outdoor facade) were removed to keep the map indoor-only and avoid mapping issues. The shipped `output` folder already reflects this; you shouldn’t need to rerun mapping (it takes ~1 night).
