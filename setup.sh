# create virtual enviroment
virtualenv env

# activate virtual enviroment
source env/bin/activate

# install packages
pip3 install paddlepaddle
pip3 install keras-ocr
pip3 install tensorflow
pip3 install craft-text-detector

# export requirements
pip3 freeze > requirements.txt
