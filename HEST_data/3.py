import openslide

for sid in ["TENX73", "TENX31", "TENX30"]:
    path = f"./hest_gbm_samples/wsis/{sid}.tif"
    slide = openslide.OpenSlide(path)

    print("=" * 60)
    print(sid)
    print("level_count:", slide.level_count)
    print("level_dimensions:", slide.level_dimensions)
    print("level_downsamples:", slide.level_downsamples)
    print("properties:")
    for k, v in slide.properties.items():
        if "mpp" in k.lower() or "aperio" in k.lower() or "openslide" in k.lower() or "objective" in k.lower():
            print(k, ":", v)