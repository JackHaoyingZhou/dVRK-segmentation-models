# Steps for real-data annotation

## Data annotation tool
Tool to annotate data: <https://github.com/haochenheheda/segment-anything-annotator>. Requires a GPU of at least 15GB to run SAM and the video tracking network.

Data annotations are saved in the labelme json format. To extract the which the classes are included in a json file run 

```bash
python /path-to-dVRK-segmentation-models/scripts/labelme_json_dataset_parsing.py parse-file image_00300.json --outdir sample_annotations/
```

This will generate a folder with the same name as the json file and inside it will have the rgb image and list of all annotations. 

## Dataset generation
To generate the final dataset, you will need to create a yaml file where you specify the color you want to use for each of the classes in your json. It is important that then the name of the classes in the yaml file matches the json file. For instance if you have a file with the annotations "background", "psm" and "phantom", your configuration yaml could look like this:

```yaml
object_names:
    - background
    - psm
    - phantom

background:
    class_id: 0
    rgb: [0,0,0]

psm:
    class_id: 1
    rgb: [0,255,0]

phantom:
    class_id: 2
    rgb: [255,0,0]
```

Using this config you can run the following command to generate the dataset:

```bash
python /path-to-dVRK-segmentation-models/scripts/labelme_json_dataset_parsing.py parse-folder --indir T1_JSON_annotations/ --outdir dataset --labels_yaml_path full_dataset.yaml
```

## Steps to annotate a video recorded with the endoscope
1. Collect video use stereo_video ros collection tool.
2. Split video into left and right
3. Sample video into images.
4. Annotate images with annotation tool
5. Convert annotations from json to BOP format

