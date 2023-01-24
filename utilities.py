# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:00:18 2022

@author: jun.yan
"""

import datetime, openai, time
import pandas as pd
from tqdm import tqdm
from io import StringIO
from config import UserConfig, DevConfig
from youtube_transcript_api import YouTubeTranscriptApi

# get transcript from youtube using the api
def get_youtube_transcript(video_id, languages=['en']):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    return transcript

# read the Otter transcript text into a list of dictionary
def get_otter_transcript(otter_file, stringio=True):
    if stringio:
        # To convert to a string based IO:
        stringio = StringIO(otter_file.getvalue().decode("utf-8"))
        # To read file as string:
        text = stringio.read()
    else:
        with open(otter_file, 'r') as f:
            text = f.read()
    text = delete_watermark(text)
    text_list = text.split('\n')
    transcript, collect = [], []
    for strs in text_list:
        strs = strs.strip()
        if strs == '':
            try:
                transcript.append({
                    'text': collect[1],
                    'start': hms_to_sec(collect[0].split()[-1]),
                })
                collect = []
            except:
                pass
        else:
            collect.append(strs)
    return transcript

# turn a transcript in dictionary format into a dataframe
def transcript_to_df(transcript):
    df = pd.DataFrame(transcript)
    df = df.sort_values('start').reset_index(drop=True)
    duration = df['start'].shift(-1) - df['start']
    duration.iloc[-1] = 0
    assert ~duration.isna().all(), 'Duration mismatch. Check the duration column.'
    assert (duration >= 0.0).all(), 'Negative duration. Check the duration column.'
    df['duration'] = duration 
    return df

# convert "hh:mm:ss" to seconds
def hms_to_sec(hms):
    result = []
    strs = reversed(hms.split(':'))
    for idx, val in enumerate(strs):
        tmp = int(val)* 60**int(idx)
        result.append(tmp)
    return sum(result)

# clean the watermarks of the text
def delete_watermark(text, watermarks=['Transcribed by https://otter.ai']):
    for w in watermarks:
        text = text.replace(w, '')
    return text

# given start-end time stamp, extract the text dataframe
def extract_df(df, start, end=''):
    if end is None or end == '':
        mask = df['start'] > hms_to_sec(start) 
    else:
        mask = (df['start'] > hms_to_sec(start)) & (df['start'] < hms_to_sec(end))
    return df[mask].reset_index(drop=True)

# use GPT-3 to produce a summary of input text
def summarize_text(text, gpt3_setting, prompt='. Summarize the text before:'):
    text = text + prompt
    response = openai.Completion.create(prompt=text, **gpt3_setting)
    return response['choices'][0]['text'].strip()

# takes a dataframe, chunk the text, and generates summary
def get_summary(df, chunk_len, word_count_col, to_summarize_col, gpt3_setting):
    tmp = df.copy()
    tmp['chunk_id'] = tmp[word_count_col].cumsum() // int(chunk_len)
    text_chunks, text_summary = [], []
    
    for cid in tqdm(tmp['chunk_id'].unique()):
        if gpt3_setting['model'] != "text-davinci-003":
            time.sleep(0.5) # to avoid hitting data transmit cap
        text = tmp[tmp['chunk_id'] == cid]['text']
        text = ' '.join(text.tolist())
        text_chunks.append(text)
        
        text = tmp[tmp['chunk_id'] == cid][to_summarize_col]
        text = ' '.join(text.tolist())
        text = ' '.join(text.split())
        text_summary.append(summarize_text(text, gpt3_setting))

    return text_chunks, text_summary, tmp

# turn results into a dataframe
def get_summary_df(df, text_chunks, text_summary):
    tmp = pd.DataFrame()
    tmp['text'] = text_chunks
    tmp['summary'] = text_summary
    tmp['start'] = df.groupby('chunk_id')['start'].head(1).reset_index(drop=True)
    tmp['duration'] = df.groupby('chunk_id')['duration'].sum().reset_index(drop=True) 
    tmp['end'] = tmp['start'] + tmp['duration']
    tmp['start_time'] = tmp['start'].apply(
        lambda x: str(datetime.timedelta(seconds=int(x)))
    )
    tmp['end_time'] = tmp['end'].apply(
        lambda x: str(datetime.timedelta(seconds=int(x)))
    )
    tmp['num_words'] = df.groupby('chunk_id')['num_words'].sum().reset_index(drop=True)
    return tmp

# write the text_chunks and summaries to file
def write_output(summary_df, summary_only=True, file_name='output.txt'):
    output = ''
    for _, row in summary_df.iterrows():
        text, summary = row['text'], row['summary']
        duration, start, end = row['duration'], row['start_time'], row['end_time']   
        if summary_only:
            output += f'*start: {start} | end: {end} | duration: {duration:.2f}s  \n'
            output += summary + '  \n\n'
        else:
            output += '***TRANSCRIPT***  \n'
            output += f'start: {start} | end: {end} | duration: {duration:.2f}s  \n'
            output += text + '  \n'
            output += '***SUMMARY***  \n'
            output += summary + '  \n\n'

    if file_name is not None:
        with open(file_name, 'w', encoding="utf-8") as f:
            f.write(output)

    return output

# source: https://stackoverflow.com/questions/72294775/
# how-do-i-know-how-much-tokens-a-gpt-3-request-used

if __name__ == "__main__":
    
    dev_mode = input("Development mode? (y or n):  ")
    setting = DevConfig if dev_mode in ['y', 'Y'] else UserConfig
    api_key = setting.api_key
    video_id = setting.video_id
    otter_file = setting.otter_file
    start_time = setting.start_time
    end_time = setting.end_time
    chunk_len = setting.chunk_len
    output_name = setting.output_name
    gpt3_setting = setting.gpt3_setting
    
    openai.api_key = api_key

    print('Select transcript source:')
    print('1 - Transcript is from YouTube. (Require a YouTube video id.)')
    print('2 - Transcript is from Otter. (Require a Otter transcript file.)')
    selection = input("Select 1 or 2:  ")
    print(f"You entered {selection}.")
    if selection not in ["1", "2"]:
        raise "Please select 1 or 2."
    if selection == "1":
        transcript = get_youtube_transcript(video_id)
    if selection == "2":
        transcript = get_otter_transcript(otter_file, False)
    
    df = transcript_to_df(transcript)
    df = extract_df(df, start_time, end_time)
    df['summary'] = df['text']
    
    i = 0
    while True:
        
        i += 1
        df['num_words'] = df['summary'].apply(lambda x: len(x.split()))
        text_chunks, text_summary, df = get_summary(
            df, 
            chunk_len, 
            'num_words', 
            'summary',
            gpt3_setting
        )
        df = get_summary_df(df, text_chunks, text_summary)
        
        now = datetime.datetime.now().strftime("%H%M%S")
        df.to_csv(f'./outputs/{output_name}_{i}_{now}.csv', index=False)
        output_text = write_output(
            df, 
            summary_only=False, 
            file_name=f'./outputs/{output_name}_{i}_{now}.txt'
        )
        output_text = write_output(
            df, 
            file_name=f'./outputs/{output_name}_{i}_short_{now}.txt'
        )
        print(output_text)
        
        run = input("Condense the summary more? (y or n):  ")
        if run not in ["Y", "y"]:
            break
    
    