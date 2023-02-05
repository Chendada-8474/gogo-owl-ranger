# Go Go Owl Ranger
A simple tool for training your custom audio samples and predicting base on CRNN architecture. It can predict probabiliy squences from each single window of spectrogram. It is suitable for detecting certain target signal from passive acoustic monitoring.

## Support
### 綠界科技
<a href="https://p.ecpay.com.tw/0696F33"><img src="https://www.ecpay.com.tw/Content/images/logo_pay200x55.png"/></a>
<br>
https://p.ecpay.com.tw/0696F33

### PayPal
<!-- PayPal Logo -->
<tr><td align="center"></td></tr><tr><td align="center"><a href="https://paypal.me/tachihchen" title="tachihchen" onclick="javascript:window.open('https://www.paypal.com/tw/webapps/mpp/paypal-popup?locale.x=zh_TW','WIPaypal','toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=1060, height=700'); return false;"><img src="https://www.paypalobjects.com/webstatic/en_US/i/buttons/pp-acceptance-medium.png" alt="使用 PayPal 立即購" /></a></td></tr>
<!-- PayPal Logo -->
<br>
https://paypal.me/tachihchen

## install

### git clone
Clone repo and install requirements.txt
```
git clone https://github.com/Chendada-8474/gogo-owl-ranger.git
cd gogo-owl-ranger
```
### Package
```
pip install -r requirements.txt
```

### GPU Accelerated Computing
For GPU acceleration, please install the compatible torch package with your device. see [INSTALL PYTORCH](https://pytorch.org). Futhermore, installing [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) correctly is also necessary.

## Documentation

### Training Custom Data

#### file structure
Create a folder under the `datasets`. Name it as any name you want. Then create the file structure as below:

```
gogo-owl-ranger
 ├ datasets
 │  ├ train
 │  │  ├ annotaion  # training annotation txt files here
 │  │  └ audio      # training audio wav files here
 │  └ val
 │     ├ annotaion  # validation annotation txt files here
 │     └ audio      # validation audio wav files here
 ├ train.py
...
```

#### annotation
-[Audacity Logo](https://www.audacityteam.org/wp-content/themes/wp_audacity/img/logo.png)
Go Go Owl Ranger eats the annotation of free, open source [Audacity](https://www.audacityteam.org/download/). Please download Audacity before tagging.

##### 1. Spectrogram Representation
Show audio as spectrogram. You also can set the *spectrogram representation* as default. edit -> preference -> track

-[select_spec](./imgs/select_spec.png)

##### 2. Creating Labels
select a region of target in the spectrogram and then press Ctrl + B.
-[create_label](./imgs/creat_label.png)

##### 3. Export Labels
copy the audio file name (without extension)
-[copy_filename](./imgs/copy_filename.png)

go to File -> Export -> Export Labels.
-[export_labels](./imgs/export_labels.png)

use the file name we just copied as the new txt file name
-[creat_label](./imgs/tag_filename.png)

Then an annotation for an audio file done!

#### training
Set the parameter in `utils` / `config.py`. Set the `dataset` as the folder name of your custom dataset and the `model_name` as you want. Adjust appropriately the `epochs` and `batch_size` according to your device and dataset.

**if you don't 100% sure the consequence of adjusting any other parameter, just don't touch it**

``` python
training_config = {
    "epochs": 20,                       # number of training epoch
    "learning_rate": 0.0005,
    "batch_size": 16,
    "dataset": "gogo-owl",              # traning dataset folder name
    "cpu_workers": cpu_count() // 2,
    "skip_false_rate": 0.7,             # The probability skip the training window if all annotations are 0
    "model_name": "grassowl",           # folder name of training outpout
}
```

### Predict

```
python3 predict.py -m models/yourmodel/best.pth -s path/to/your/data -b 16
```

| argument      | abbreviation | default | require | description |
| ------------- | ------------ | ------- | ------- | ----------- |
| `--model`     | `-m`         |         | true    | model path  |
| `--batch`     | `-b`         |         | true    | batch size  |
| `--source`    | `-s`         |         | true    | target wav file or folder |
| `--interval`  | `-i`         | 0.5     |         | the interval (s) of probability sequence results, the probability is the maximun of all the origin sequence from model in an interval |
| `--threshold` | `-t`         | 0.5     |         | probability threshold of presence or absence of target sound (when caculating the `target_coverage.csv`), range 0 ~ 1 |

results are saved as `probability_sequence.csv` and `target_coverage.csv` under the folder of source files.