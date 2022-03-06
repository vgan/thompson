#!/usr/bin/env python
from subprocess import Popen,PIPE,STDOUT
from mastodon import Mastodon
import tweepy
import os,mmap

folkmotifcmd = [ "/usr/local/bin/python","/tf/thompson/eval.py","--gen_len=2000","--save_path=/tf/thompson","--temperature=1" ]
p = Popen(folkmotifcmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
output = p.stdout.read()
lines = output.splitlines()

# import Mastodon creds from env vars
client_key = os.environ['CLIENT_KEY']
client_secret = os.environ['CLIENT_SECRET']
access_token = os.environ['ACCESS_TOKEN']
# import twitter creds from env vars
twitter_consumer_key = os.environ['TWITTER_CONSUMER_KEY']
twitter_consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
twitter_token_key = os.environ['TWITTER_TOKEN_KEY']
twitter_token_secret = os.environ['TWITTER_TOKEN_SECRET']

mastodon = Mastodon(
    api_base_url='https://botsin.space',
    client_id=client_key,
    client_secret=client_secret,
    access_token = access_token
)

basedir = "/tf/thompson/"
past_motifs = basedir + "past_motifs.txt"
i = 50

def pick_motif(i):
    motif = lines[i].lstrip()
    motif = motif.capitalize()
    chars = len(motif)
    return motif

def isItNew(motif):
    if os.path.isfile(past_motifs):
        past_motifsObj = open(past_motifs,"a+")
        past_motifsmm = mmap.mmap(past_motifsObj .fileno(), 0, access=mmap.ACCESS_READ)
        if past_motifsmm.find(motif) == -1: # text not found, string is new!
            past_motifsObj.write(motif.decode() + str("\n")) # store text to check for uniqueness
            unique = True
            return unique
        else:
            unique = False
            return unique
        past_motifsObj.close()
    else:
        print(past_motifs + " does not exist...")

def qualityControl(motif_b,i):
    if (b"rape" in motif_b):
        print("skipping motif involving rape")
        i = i + 1
        motif_b = pick_motif(i)
    if motif_b[-1:] != b".":
        print("Skip if it doesnt end in period")
        i = i + 1
        motif_b = pick_motif(i)
    if len(motif_b) <= 10:
        print("Too short!")
        i = i + 1
        motif_b = pick_motif(i)
    return(motif_b,i)

motif_b = pick_motif(i)
motif = motif_b.decode()

qualitycontrol = qualityControl(motif_b,i)
motif_clean = qualitycontrol[0]
i = qualitycontrol[1]
motif = motif_clean.decode()

while isItNew(motif_b) == False:
    print("Dupe, trying another one.")
    i = (i+1)
    motif = pick_motif(i)
    motif = motif.decode()
    qualitycontrol = qualityControl(motif,i)
    motif_clean = qualitycontrol[0]
    motif = motif_clean.decode()

try:
    auth = tweepy.OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
    auth.set_access_token(twitter_token_key, twitter_token_secret)
    api = tweepy.API(auth)
    tweet = motif
    status = api.update_status(status=tweet)
except:
    print("twete failed :(")
try:
    toot_text = motif
    tooterino = mastodon.status_post(toot_text, sensitive=False)
except:
    print("toot failed")

print(motif)
