# Label-Studio-Flow
## Description

This script is intented to implement the Human in the Loop pipeline for the NER annotation task using the [Label Studio](https://labelstud.io/) labeling tool.

## Usage 

```
$ pyhton main.py <config_file>
```

## Config File 
The configuration file provided to this script must contain the sections `SETTINGS` and `MODELS`

### `SETINGS` Section

The `SETINGS` section sets preferences about the Label Studio and Machine Learning interfaces. The following options must be provided:

|Option|Description|
|------|-----------|
|`label_studio_url`|Url for the Label Studio server|
|`label_studio_api_key`|API key for the Label Studio server|
|`labeled_project_id`| ID of the project that contains data that will be used in the training process|
|`lunabeled_project_id`| ID of the project that will handle the annotation process|

### `MODELS` Section
The `MODELS` section defines the models to be used in the training process. To do that, you just need to provide an arbitrary name for your model and a path for it in the [Hugging Face](https://huggingface.co) website. For exemple:

```
[MODELS]
BERTimbau=neuralmind/bert-base-portuguese-cased
BERTLener=pierreguillou/bert-base-cased-pt-lenerbr
```

### Example
```
[USER]
label_studio_url=http://myserver.com/labelstudio
label_studio_api_key=1212312434
labeled_project_id=1
unlabeled_project_id=2

[MODELS]
bert=bert-base-multilingual-cased
bertimbau=neuralmind/bert-base-portuguese-cased
bertlener=pierreguillou/bert-base-cased-pt-lenerbr
```


