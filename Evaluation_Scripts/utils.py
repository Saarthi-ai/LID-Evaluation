import glob
import subprocess
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from models import *
from pydub import AudioSegment
from pydub.silence import split_on_silence
# import json
# import string
import time
# import timeit
# import threading
# from tracemalloc import start
# from threading import Timer
# from typing import final
# import wave
# import time
# from base64 import b64decode
# from scipy.io.wavfile import read, write
# from scipy.io import wavfile
# from email.mime import audio
# import io

ACCESS_TOKEN = config('ACCESS_TOKEN', cast=str)

def data_loading(path_to_df):

    # Load and clean dataset
    print('Loading Dataset....')
    df = pd.read_csv(path_to_df)
    df.dropna(inplace = True)
    df.reset_index(drop = True, inplace = True)

    return df

def pruned_dataframe(df):

    # Choose 4 most abundant languages
    print('Pruning DataFrame....')
    lang_dict = Counter(df['information.language'])
    sorted_by_counts = list(sorted(lang_dict.items(), key = lambda x: x[1], reverse = True))
    top_4_lang = [ i[0] for i in sorted_by_counts ][:4]

    # Eliminate other languages and eliminate unwanted features
    sub_df=pd.DataFrame(columns=['id', 'language', 'identified'])
    idx=0   # Index for sub_df

    # Select ID, language from original df and assign identified language as None
    for i in range(len(df)):
        if df.loc[i]['information.language'] in top_4_lang:
            sub_df.loc[idx]=[df.loc[i]['information.sessionId'], df.loc[i]['information.language'], None]
            idx+=1

    return sub_df


def runcmd(cmd, verbose = False, *args, **kwargs):

    # Run bash commands
    process = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True, shell = True)
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def download_recordings(df, audio_save_dir, utterances_save_dir, clipped_audio_dir):

    # Create dir for audio and customer utterances
    runcmd(cmd = 'mkdir {}'.format(audio_save_dir))
    runcmd(cmd = 'mkdir {}'.format(utterances_save_dir))
    runcmd(cmd = 'mkdir {}'.format(clipped_audio_dir))


    # wget files from links provided in DataFrame
    print('Downloading Files....')
    for i in df['information.recording_file_url']: # pass original dataframe
        cmd = f'wget -O {os.path.abspath(audio_save_dir)}/{i[58:94]}.mp3 \"{str(i)}?{ACCESS_TOKEN}\"'     # Change the name of the output file
        runcmd(cmd=cmd)


    # Include /path/to/dir/* at the end. Required for glob.glob
    if audio_save_dir[-1] != '*':
        audio_save_dir=os.path.abspath(audio_save_dir)+'/*'

    # Split the dual channel audio; Save only the left channel (customer utterance)
    files_list = glob.glob(audio_save_dir)
    print('Splitting Customer Utterances from audio....')
    for i in files_list:
        cmd='ffmpeg -i \"'+i+'\" -map_channel 0.0.0 \"'+os.path.join(utterances_save_dir, i.split('/')[-1].split('.mp3')[0])+'_customer.wav\"'
        runcmd(cmd)

        
    # Include /path/to/dir/* at the end. Required for glob.glob
    if utterances_save_dir[-1] != '*':
        utterances_save_dir = os.path.abspath(utterances_save_dir)+'/*'
        
    # Clip out the silent regions from audios
    utterance_list = glob.glob(utterances_save_dir)
    print('Clipping silent portions out of audio files....')
    for i in utterance_list:
        sound = AudioSegment.from_file(i, format = 'wav')
        audio_chunks = split_on_silence(sound, min_silence_len = 100, silence_thresh = -45, keep_silence = 50)
        combined = AudioSegment.empty()
        
        for chunk in audio_chunks:
            combined += chunk
        combined.export('{0}/{1}'.format(os.path.abspath(clipped_audio_dir), i.split('/')[-1]), format = 'wav')
        


def make_predictions(model_id, utter_dir, sub_df):
    
    model = {0: speech_language_detection_once_from_file, 1: huggingface_LID_model}
    res_file = {0: 'results_azure.csv', 1: 'results_hf.csv'}
    print('Making Predictions....')
    
    count_not_ided = 0
    t1 = time.perf_counter()
    
    for i in range(len(sub_df)):
        if not i % 100: print('Sample {}'.format(i))
        path=os.path.join(utter_dir, sub_df.loc[i]['id'])+'_customer.wav'
        try:
            sub_df.loc[i]['identified']=model[model_id](path)
        except :
            sub_df.loc[i]['identified']=4   # couldn't identify due to exceptions
            count_not_ided += 1
    
    t2 = time.perf_counter()
    
    total = len(sub_df)-count_not_ided
    print('Time taken for evaluation of {} samples is {:.2f} seconds.\nLatency is: {:.2f}'.format(total, t2-t1, ((t2-t1)/total)))
    
    if model_id: runcmd('rm *.wav') # Model creates copies of wav files in the pwd which must be deleted
    
    # Convert Languages to IDs
    label_to_Int = {'English':0, 'Hindi':1, 'Tamil':3, 'Kannada':3} if model_id else {'English':0, 'Hindi':1, 'Tamil':2, 'Kannada':3}   # make changes if newer models are added
    for i in range(len(sub_df)):
        sub_df.loc[i]['language']=label_to_Int[sub_df.loc[i]['language']]
        
    # Save results
    sub_df.to_csv(res_file[model_id])


def evaluate(model_id):
    
    res_file = {0: 'results_azure.csv', 1: 'results_hf.csv'}
    sub_df = pd.read_csv(res_file[model_id])
    
    yhat = sub_df['identified']
    y = sub_df['language']
    
    # Unrecognized Percentage
    res_count_dict = Counter(yhat)
    try:
        print('Percent unrecognized: {:.2f}'.format(res_count_dict[4]/sum(res_count_dict.values())*100))
    except ZeroDivisionError as e:
        print(e)
        pass

    # Accuracy including Not Identified
    correct_not_det=0
    for i in range(len(yhat)):
        if yhat[i] == y[i]:
            correct_not_det += 1

    print('Accuracy including not identified: {:.2f}'.format(correct_not_det/len(y)*100))

    # Remove not identified from sub_df
    sub_df_det = sub_df[sub_df.identified != 4]
    sub_df_det.reset_index(drop=True, inplace=True)

    # Accuracy excluding Not Identified
    correct_det = 0
    for i in range(len(sub_df_det)):
        if sub_df_det.identified[i]==sub_df_det.language[i]:
            correct_det+=1

    try:
        print('Accuracy excluding not identified: {:.2f}'.format(correct_det/len(sub_df_det)*100))
    except ZeroDivisionError as e:
        print(e, 'Variable sub_df_det: {} may be empty'.format(sub_df_det), sep='\n')
        pass
    
    # Confusion Matrix
    if model_id:
        cm = confusion_matrix(y, yhat, labels=[0,1,3,4])
        cm_df = pd.DataFrame(cm, index = ['English','Hindi','Other', 'None'],
            columns = ['English','Hindi','Other', 'None'])
    else:
        cm = confusion_matrix(y, yhat, labels=[0,1,2,3,4])
        cm_df = pd.DataFrame(cm, index = ['English','Hindi','Tamil', 'Kannada', 'None'],
                columns = ['English','Hindi','Tamil', 'Kannada', 'None'])
    print(cm_df)

    # # Plot as Heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
