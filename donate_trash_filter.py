import os, glob
import numpy as np
from tqdm import tqdm
import librosa
from tensorflow.keras.models import Model, load_model


# config.
SAMPLING_RATE = 16000
SEGMENT_LENGTH = 3
TOTAL_AUDIO_LENGTH_SEC = 60
BATCH_SIZE = 10

# constants.
FRAME_SAMPLES = SAMPLING_RATE * SEGMENT_LENGTH
FIX_TOTAL_AUDIO_SAMPLE = SAMPLING_RATE * TOTAL_AUDIO_LENGTH_SEC



assert (FIX_TOTAL_AUDIO_SAMPLE / FRAME_SAMPLES) %  BATCH_SIZE == 0, "Total audio sample should be divisible by frame samples."




def load_audio_track(file_path):
    track, _= librosa.load(file_path, sr=SAMPLING_RATE, mono=True, res_type='kaiser_fast')
    if len(track) < FIX_TOTAL_AUDIO_SAMPLE:
        pad_len = FIX_TOTAL_AUDIO_SAMPLE - len(track)
        track = np.concatenate([track, np.zeros(pad_len)])
    else:
        track = track[:FIX_TOTAL_AUDIO_SAMPLE]

    return track



def encode_template(template_track):
    # pre-encode template track to [20, 60, 9, 192]
    template_track = template_track.reshape((-1, BATCH_SIZE, FRAME_SAMPLES))

    template_feats = []
    for b in template_track:
        template_feats.append(track1_encoder.predict(b))
    return np.stack(template_feats)


def compare(template_feats, target_audio):
    target_audio = target_audio.reshape((-1, BATCH_SIZE, FRAME_SAMPLES))

    result = []
    for tem, tar in zip(template_feats, target_audio):
        result.append(model.predict([tem, tar]))

    return np.average(result)






TEMPLATE_PATH = "data\\anchor\\anchor.mp3"
DONATE_AUDIOS = "data\\production\\*.mp3"
# DONATE_AUDIOS = "data\\production\\f85e803c-09fb-4ccc-a86b-0085eff83e04.mp3"



# load models.
track1_encoder = load_model("models\\siam_sim_track1_encode_b10.h5")
model = load_model("models\\siam_sim_b10_deploy.h5")



if __name__ == '__main__':


    # load template track.
    template_track = load_audio_track(TEMPLATE_PATH)
    template_feats = encode_template(template_track)


    # donate files.
    donate_files = glob.glob(DONATE_AUDIOS)


    for target_file in tqdm(donate_files):

        target_audio = load_audio_track(target_file)
       
        score = compare(template_feats, target_audio)

        if score < 0.72 : 
            file_name = os.path.split(target_file)[-1]
            target_location = os.path.join("data", "trash", file_name)
            os.rename(target_file, target_location)

    # target_audio = load_audio_track(DONATE_AUDIOS)
       
    # score = compare(template_feats, target_audio)

    # print('Score: ', score)


