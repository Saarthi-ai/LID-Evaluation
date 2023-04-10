from utils import *

if __name__ == '__main__':

    path = './dataset.csv'
    audio_dir = './audios'
    utterance_dir = './utterances'
    clipped_audio_dir = './clipped_audios'
    df = data_loading(path)
    sub_df = pruned_dataframe(df)
    download_recordings(df, audio_dir, utterance_dir, clipped_audio_dir)
    model_id = int(input("Enter 0 for Azure \n1 for HuggingFace \n"))    
    make_predictions(model_id, clipped_audio_dir, sub_df)
    evaluate(model_id)
    