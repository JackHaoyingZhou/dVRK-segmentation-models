# labelme processing scripts

## Parse a single json labelme file
```
python scripts/labelme_json_dataset_parsing.py parse-single-json-file --help
```

```
Usage: labelme_json_dataset_parsing.py parse-single-file 
           [OPTIONS] JSON_FILE

  Parse a single labelme json file and extract raw image and label

Options:
  --outdir PATH            [required]
  --labels_yaml_path PATH  Path to yaml file with class names and ids
  --help                   Show this message and exit.
```

## Parse folder of json labelme files

```
python scripts/labelme_json_dataset_parsing.py parse-folder --help
```

```
Usage: labelme_json_dataset_parsing.py parse-folder [OPTIONS]

Options:
  --indir PATH             Path to the folder with json files  [required]
  --outdir PATH            [required]
  --labels_yaml_path PATH  Path to yaml file with class names and ids
                           [required]
  --help                   Show this message and exit.
```