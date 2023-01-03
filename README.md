# ocr-exp

**Note**

- `paddleocr` required upper python v3.7
- `numpy` required version < 1.24.0 (==1.21.0)
- uninstall all python packages `pip freeze | xargs pip uninstall -y`

# packages

- opencv `pip3 install opencv-python`

- pyzbar `pip3 install pyzbar` (unbutu 18.04.06 need to install `sudo apt-get install -y libzbar0` or `apt-get install zbar-tools`)

- openvino-dev `pip install openvino-dev` (upgrade pip before installing `python -m pip install --upgrade pip` or in container `python3 -m pip install --upgrade pip`)

- paddle `pip install paddlepaddle`

- shapely `pip install shapely`

- pyclipper `pip install pyclipper`

- jupyter `pip install jupyter`

- matplotlib `pip install matplotlib`

-circle_fit `pip install circle-fit`

# references

[model_list_en](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)

[paddleocr](https://pypi.org/project/paddleocr/)

[paddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

[paddleOCR doc](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/doc)

[paddleOCR deploy](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/deploy)
