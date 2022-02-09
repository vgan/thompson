# Char RNN using TensorFlow 2.0 via Docker

## Based on Official [TF2 Text Generation Tutorial](https://www.tensorflow.org/text/tutorials/text_generation)

## Trained on some cleaned up text from the [Stith Thompson's Motif-Index of Folk-Literature](https://archive.org/details/Thompson2016MotifIndex)

Build and tag the docker image to run locally
`docker build . --tag thompson:v1.0.0`

Start the Docker image using a local volume and the auth data for Mastodon and Twitter
`docker run --name thompson --rm --gpus all -it -v "thompson:/tf/thompson" --env CLIENT_KEY="..." --env CLIENT_SECRET="..." --env ACCESS_TOKEN="..." --env TWITTER_CONSUMER_KEY="..." --env TWITTER_CONSUMER_SECRET="..." --env TWITTER_TOKEN_KEY="..." --env TWITTER_TOKEN_SECRET="..." thompson:v1.0.0 "/tf/thompson/rnn_folkmotif.py"`

The past_motifs.txt file is used to store previously generated motifs to check for uniqueness. Newly generated motifs are appended to the file.

Starting a container and launching a shell to train or sample data manually.
`docker run --name thompson --rm --gpus all -it -v "thompson:/tf/thompson" thompson:v1.0.0 bash`

Return text sample output from the model. The higher the temperature number the weirder the results get.
`python eval.py --gen_len=2000 --save_path=/tf/thompson --temperature=1`

Note: Not required, but if you want to retrain using the source input.txt or train a new model using your own text.
`python train.py --data_path=/tf/thompson/input.txt --save_path=/tf/thompson --epochs=500 --n_layers=3 --n_embedding=128`

Working examples running the bot can be found here:
  [Twitter](https://twitter.com/neuralfolk) and [Mastodon](https://botsin.space/@neuralfolkmotifs)  
