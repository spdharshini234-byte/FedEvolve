# Data folder

```
data/raw/            <- YOUR images (one sub-folder per defect class)  [not included]
    scratch/  crack/  dent/  color_defect/   (names must match `dataset.classes` in configs/config.yaml)
data/processed/      <- created by `python run.py prepare-data` (resized train/val/test + CSV lists)
data/clients/        <- created by `python run.py create-clients` (one CSV list per simulated factory)
```

* Supported formats: jpg, jpeg, png, bmp, tif, tiff, webp. At least 10 images per class (more is better).
* No real dataset is included. To try the software without one, run `python run.py prepare-demo`:
  it draws **synthetic** images (not real defects) into `data/raw/` and marks them with a `.demo_dataset` file.
  Results from the demo dataset only prove that the pipeline works.
* Real dataset ideas (download yourself, check licenses): NEU Surface Defect Database, MVTec AD, Kolektor SDD.
  Arrange them into one folder per class and edit `dataset.classes` and the `continual.tasks` / `non_iid.class_weights` in the YAML files.
* Factories do not get separate copies of images. Each client folder holds a CSV list of the images that factory "owns";
  the code only ever sends model weights to the server.
