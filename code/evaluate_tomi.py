# Author: Shawn Lee
# Date: Aug 2023
# Description: Evaluation code to evaluate ToMI.

import random
import tqdm
import argparse
import prompts_tomi
from sim_utils_tomi import *
import collections
import json
from huggingface_hub import login


DATA_DIR = '../data/fixedtomi/test_balanced.jsonl'
OUTPUT_DIR = './outputs'
random.seed(0)

question_types = [
'first_order_0_no_tom',
'first_order_1_no_tom',
'first_order_0_tom',
'first_order_1_tom',
'second_order_0_no_tom',
'second_order_1_no_tom',
'second_order_0_tom',
'second_order_1_tom',
'memory',
'reality',
]

story_types = ['true_belief','false_belief','second_order_false_belief']

def most_common(lst):
    data = collections.Counter(lst)
    return data.most_common(1)[0][0]

def read_txt(filename):
    with open(filename, 'r') as f:
        s = f.read()
    return s

def evaluate_tomi():
    # Load the dataset
    gold = []             # Ground truths
    predictions = []      # Predictions
    results = []          # Results (accuracy)
    category_results = {} # Results per category
    category_percents = {}
    story_results = {}
    story_percents = {}
    bad = 0
    correctNum = 0
    totalNum = 0

    if args.wandb == 1:
        run = wandb.init(
            project=args.project,
            entity=args.entity, 
            config=vars(args),
            tags=args.tags.split(','),
        )

        table = wandb.Table(columns=["story", "perspective", "question", "story_type", "question_type", "label", "guess","correct"])
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    
    # Print stuff
    print("\n------------------------")
    print("    EVALUATING TOMI      ")
    print("------------------------")
    if args.eval_model == None:
        print(f"PER MODEL: {args.perspective_model}")
        print(f"SIM MODEL: {args.sim_model}")
    else:
        print(f"EVAL MODEL: {args.eval_model}")
    print(f"DATA: {args.data_dir}")
    print(f"METHOD: {args.method}")
    print(f"CATEGORY: {args.category}")
    print(f"N = {args.num_probs}")
    print("------------------------\n")

    if 'Llama' in args.eval_model:
        os.environ["HF_TOKEN"] = ''
        login(os.environ["HF_TOKEN"])

    # Put save path
    model_id = args.eval_model.split('/')[-1]
    save_path = os.path.join(OUTPUT_DIR, f'reasoning_tomi_{args.method}_{model_id}.txt')

    with open(save_path, "a") as f:
        f.write("---------EVALUATING WITH--------\n")
        if args.eval_model == None:
            f.write(f"PER MODEL: {args.perspective_model}\n")
            f.write(f"SIM MODEL: {args.sim_model}\n")
        else:
            f.write(f"EVAL MODEL: {args.eval_model}\n")
        f.write(f"DATA: {args.data_dir}\n")
        f.write(f"METHOD: {args.method}\n")
        f.write(f"CATEGORY: {args.category}\n")
        f.write(f"N = {args.num_probs}\n")
        f.write("------------------------\n\n")

    
    if args.eval_model == None:
        if "gpt" in args.perspective_model:
            perspectiveModel = ChatGPT(args.perspective_model, temperature=args.temperature, verbose=args.verbose, save_path=save_path)
        else:
            perspectiveModel = LLM(args.perspective_model,load_in_8bit=args.eight_bit, temperature=args.temperature, verbose=args.verbose, do_sample=args.do_sample, save_path=save_path)
        
        if "gpt" in args.sim_model:
            simModel = ChatGPT(args.sim_model, temperature=args.temperature, verbose=args.verbose, save_path=save_path)
        else:
            simModel = LLM(args.sim_model,load_in_8bit=args.eight_bit, temperature=args.temperature, verbose=args.verbose, do_sample=args.do_sample, save_path=save_path)

    else:
        simModel = None
        if "gpt" in args.eval_model:
            perspectiveModel = ChatGPT(args.eval_model, temperature=args.temperature, verbose=args.verbose, save_path=save_path)
        else:
            perspectiveModel = LLM(args.eval_model,load_in_8bit=args.eight_bit, temperature=args.temperature, verbose=args.verbose, do_sample=args.do_sample, save_path=save_path)

    
    # Query model on dataset
    with open(args.data_dir) as f_in:
        i = 0
        for index, line in tqdm.tqdm(enumerate(f_in), total=min(len(read_txt(args.data_dir).split('\n')), args.num_probs)):
            # Parse each line as JSON
            fields = json.loads(line.strip())
            
            # extract fields from JSON
            story, question, label, containers, story_type, question_type = fields["story"], fields["question"], fields["answer"], fields["containers"], fields["story_type"], fields["question_type"]
            
            # If category is specified, then only evaluate that question type.
            if args.category != "all" and question_type not in args.category.split(","):
                continue
            
            i += 1
            if i > args.num_probs:
                break
            
            baselinePrompt = prompts_tomi.baselinePrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            rulesPrompt = prompts_tomi.rulesPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            cotPrompt = prompts_tomi.cotPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            rulesCoTPrompt = prompts_tomi.rulesCoTPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            rulespsPrompt = prompts_tomi.rulespsPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            questionPrompt = prompts_tomi.questionPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            questionPromptCoT = prompts_tomi.questionPromptCoT.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            
            
            if "llama" in args.sim_model:
                # We have separate prompts for these because, again, llama is stubborn.
                # Nothing too crazy, we just start off its response for it.
                baselinePrompt = prompts_tomi.llamaprompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
                cotPrompt = prompts_tomi.llamaCotPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            
            if args.method == "baseline":
                perspective = "N/A (Baseline)"
                prediction = perspectiveModel.getOutput(baselinePrompt)
                
            elif args.method == "baselineRules":
                perspective = "N/A (Baseline)"
                prediction = perspectiveModel.getOutput(rulesPrompt)
                                
            elif args.method == "cot":
                perspective = perspectiveModel.getOutput(cotPrompt)
                try:
                    prediction = perspective.split("Answer:")[1]
                except:
                    prediction = perspective[-30:]
                perspective = "Chain of Thought: \n" + perspective

            elif args.method == "cotRules":
                perspective = perspectiveModel.getOutput(rulesCoTPrompt)
                try:
                    prediction = perspective.split("Answer:")[1]
                except:
                    prediction = perspective[-30:]
                perspective = "Chain of Thought: \n" + perspective
                
            elif args.method == "psRules":
                perspective = perspectiveModel.getOutput(rulespsPrompt)
                try:
                    prediction = perspective.split("Answer:")[1]
                except:
                    prediction = perspective[-30:]
                perspective = "Chain of Thought: \n" + perspective
                
            elif args.method == "simulation" or args.method == "ptom":
                prediction, perspective = evalQuestion(perspectiveModel, fields, questionPrompt, simModel=simModel, ptom=args.method=="ptom")
            elif args.method == "simulationcot":
                prediction, perspective = evalQuestion(perspectiveModel, fields, questionPromptCoT, simModel=simModel)
                try:
                    prediction = prediction.split("Answer:")[1]
                except:
                    prediction = prediction[-30:]
            elif args.method == "ours":
                prediction, perspective = evalQuestion_ours(perspectiveModel, fields, questionPrompt, simModel=simModel)   
            elif args.method == "ours_full":
                prediction, perspective = evalQuestion_ours2(perspectiveModel, fields, questionPrompt, simModel=simModel)                             
            else:
                print("Please specify a valid evaluation method!")
                return
            
            
            # Grade answer
            correct = None
            if label in prediction.lower() or ' '.join(label.split('_')) in prediction.lower():
                correct = True
            else:
                correct = False
            
            if correct:
                correctNum += 1
            totalNum += 1
            
            

            if args.verbose:
                print(f"\n### Correct: {correct} ###\n")

            # if not correct:
            #     # This means the model got it wrong.
            if args.wandb == 1:
                table.add_data(story, perspective, question, story_type, question_type, label, prediction, correct)

            
            # Append ground truth and model prediction.
            gold.append(label)
            predictions.append(prediction)
            
            # Calculate category result
            temp = category_results.get("count"+question_type, {"correct": 0, "total" : 0})
            if correct:
                temp["correct"] += 1
            temp["total"] += 1
            percent = temp["correct"] / temp["total"]
            category_results["count"+question_type] = temp
            category_percents[question_type] = percent

            temp_story = story_results.get("count"+story_type, {"correct": 0, "total" : 0})
            if correct:
                temp_story["correct"] += 1
            temp_story["total"] += 1
            percent_story = temp_story["correct"] / temp_story["total"]
            story_results["count"+story_type] = temp_story
            story_percents[story_type] = percent_story
            
            if prediction == -1:
                bad += 1

            if args.method in ['simulation','ptom','ours','simulationcot','ours_full']:
                with open(save_path, "a") as f:
                    f.write("---------Perspective--------\n")
                    f.write(perspective + '\n\n')
                    f.write(f"graded answer: {str(correct)}" + '\n\n')
            else:
                with open(save_path, "a") as f:
                    f.write(f"graded answer: {str(correct)}" + '\n\n')
            
            if args.wandb == 1:
                try:
                    rollingAccuracy = correctNum / totalNum
                    rollingAccuracy_dict = {'running_acc' : rollingAccuracy}
                    wandb.log({**category_percents, **rollingAccuracy_dict, **category_results})
                except:
                    pass


    accuracy = correctNum / totalNum
    print(f"Accuracy: {accuracy*100:.3f}%")
    results.append(f"Accuracy : {accuracy:.3f}")
    print(f"Bad responses: {bad}")
    results.append(f"Bad responses: {bad}")

    if args.wandb == 1:
        wandb.log({'total_acc' : accuracy})
        wandb.log({'bad_responses' : bad})
        wandb.log({'wrong_answers': table})
        wandb.finish()

    # Print results
    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {args.eval_model}")
    print(f"METHOD: {args.method}")
    print(f"ACCURACY: {accuracy:.2%}")
    print("------------------------\n")
    
    with open(os.path.join(OUTPUT_DIR, f'metrics_tomi_{args.method}_{model_id}.txt'), "a") as f:
        f.write("---------RESULTS--------\n")
        f.write(f"EVAL MODEL: {args.eval_model}" + '\n')
        f.write(f"METHOD: {args.method}\n")
        f.write(f"ACCURACY: {accuracy:.2%}\n")
        f.write(f"BAD: {bad}\n\n")
        for question_type in question_types:
            p = category_percents[question_type]
            t,c = category_results["count"+question_type]['total'], category_results["count"+question_type]['correct']
            f.write(f"{question_type}: {p:.2%} ({c}/{t})\n")
        f.write("\n")
        for story_type in story_types:
            p = story_percents[story_type]
            t,c = story_results["count"+story_type]['total'], story_results["count"+story_type]['correct']
            f.write(f"{story_type}: {p:.2%} ({c}/{t})\n")

        so_fb_t = story_results["countsecond_order_false_belief"]['total']
        fb_t = story_results["countfalse_belief"]['total']
        so_fb_c = story_results["countsecond_order_false_belief"]['correct']
        fb_c = story_results["countfalse_belief"]['correct']
        fb_total = (so_fb_c + fb_c) / (so_fb_t + fb_t)
        f.write(f"false_belief (total): {fb_total:.2%} ({so_fb_c + fb_c}/{so_fb_t + fb_t})\n")
        f.write("------------------------\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/fixedtomi/test_balanced.jsonl')
    parser.add_argument('--eval_model', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_probs', '-n', type=int, default=1000)
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--eight_bit', action='store_true')
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--tags', type=str, default="debug")
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--perspective_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--sim_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--do_sample', action='store_true') # Apply sampling?
    global args
    args = parser.parse_args()
    
    print(args)

    model_id = args.eval_model.split('/')[-1]
    save_path = os.path.join(OUTPUT_DIR, f'reasoning_tomi_{args.method}_{model_id}.txt')
    if args.category=='all' and os.path.exists(save_path): os.remove(save_path)

    # Evaluate on ToMi
    evaluate_tomi()

if __name__ == '__main__':
    main()