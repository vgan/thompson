FROM tensorflow/tensorflow:latest-gpu-py3 #For nvidia gpu
WORKDIR /tf/
RUN apt update
RUN apt install -y git curl vim
RUR git clone https://github.com/vgan/thompson.git
RUN pip3 install numpy colorama
