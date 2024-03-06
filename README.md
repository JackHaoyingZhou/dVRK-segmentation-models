# Surgical segmentation models

Utility library for semantic segmentation of surgical instruments. To install the package, follow the installation instructions [here](./docs/installation.md). Additional notes about how to use the deep learning models can be found in the [model notes](./docs/model_notes.md) page. Documentation of scripts can be found [here](./docs/script_descriptions.md).

# Notes
* Monai blend function works with normalized and unnoralized images.
* Model produces logits. Last layer of Flexible Unet is a conv layer.
* Number of classes is automatically extracted from Labels config file.
* Does training depends on using a sigmoid or softmax activation function?
* Prediction transformations are probably not correct right now. Threshold should not be applied to logits.

## TODO
* [ ] Combine multiple colors into a single class in LabelParse config file. 
* [ ] Output inferences in a directory.
