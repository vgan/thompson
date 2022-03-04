FROM tensorflow/tensorflow
WORKDIR /tf/
RUN apt update
RUN apt install -y git curl vim
RUN git clone https://github.com/vgan/thompson.git
RUN chmod +x /tf/thompson/rnn_folkmotif.py
RUN pip3 install numpy colorama Mastodon.py tweepy
