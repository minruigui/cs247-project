import argparse
import os
import numpy as np
import pandas as pd
from bitsandbytes.nn import Int8Params
choices = ["A", "B", "C", "D"]
from transformers import AutoModelForCausalLM, AutoTokenizer



# prompt = "My favourite condiment is"

# model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

# generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
# tokenizer.batch_decode(generated_ids)[0]



def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
BACH_SIZE = 1
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", special_tokens_map={})
def load_quantized_model(model_name, quantization):
    if quantization == "none":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    elif quantization == "8bit":
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    elif quantization == "4bit":
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
        os.environ["BNB_4BIT_COMPUTE_DTYPE"] = "float16"
    return model
def eval(args,model, subject, dev_df, test_df):
    
    # def crop_prompt(prompt: str):
    #     cropped_prompt = tokenizer.decode(tokenizer.encode(prompt,bos=True, eos=False)[:2048])
    #     return cropped_prompt

    # def crop(s):
    #     prompt = crop_prompt(s)
    #     return prompt
    cors = []
    prompts = []
    labels=[]
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # while crop(prompt) != prompt:
        #     k -= 1
        #     train_prompt = gen_prompt(dev_df, subject, k)
        #     prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]
        prompts.append(prompt)
        labels.append(label)
        if len(prompts) == BACH_SIZE:
            model_inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            cs = tokenizer.batch_decode(generated_ids[:,-1])
            for c,l in zip(cs,labels):
                cors.append(c==l)
            prompts=[]
            labels=[]
    if len(prompts)>0:
        model_inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
        generated_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        cs=tokenizer.batch_decode(generated_ids[:,-1])
        for c,l in zip(cs,labels):
            cors.append(c==l)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc

def main(args):
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    print(subjects)
    print(args)
    all_cors = []
    model = load_quantized_model(args.model, args.quantization)
    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc = eval(args,model, subject, dev_df, test_df)
        all_cors.append(cors)
        test_df["correct"] = cors
        # for j in range(probs.shape[1]):
        #     choice = choices[j]
        #     test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, "{}.csv".format(subject)), index=None)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--quantization", "-q", type=str, default="none", choices=["none", "8bit", "4bit"], help="Quantization type")
    args = parser.parse_args()
    main(args)

