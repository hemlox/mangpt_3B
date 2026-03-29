from datasets import load_dataset
from textwrap import dedent
dataset = load_dataset("parquet", data_files="pairs.parquet",split="train")
past_5_messages = []
def generate_prompt(row):
    try:
        user_text = row['messages'][0]['content']#user
        response_text = row['messages'][1]['content']#assistant
        if len(past_5_messages) == 0:
            sliding_window = "No Preceding Messages"
        else:
            sliding_window = str(past_5_messages)
        alpaca_formatted_string = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{user_text}

### Input:
You are reading the messages of a discord server. These are the 5 messages preceding this
{sliding_window}

### Response:
{response_text}<|end_of_text|>'''#Adding EOS mad imp almost forgor
        ## updating the sliding window
        if len(past_5_messages) < 2:
            past_5_messages.append(user_text)
            past_5_messages.append(response_text)
        elif len(past_5_messages) <5:
            past_5_messages.append(response_text)
        else:
            past_5_messages.pop(0)
            past_5_messages.append(response_text)
        return {'text' : dedent(alpaca_formatted_string)}
    except:
        return  {'text' : None}
dataset = dataset.map(generate_prompt,remove_columns=['metadata', 'messages'], load_from_cache_file=False)
dataset = dataset.filter(lambda x:x['text'] is not None)
dataset.to_parquet('updated_pairs.parquet')