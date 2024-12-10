import argparse
import sys
import os
from pathlib import Path
from compose import LLM_Discussion_SciFi, LLM_Teacher_SciFi, LLM_Review_SciFi
from types import SimpleNamespace
from pathlib import Path


# This file run LLM Discussion, Teacher, and Review

def main():
    parser = argparse.ArgumentParser(description="Orchestrate a discussion with multiple AI agents.")
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file for agents.")
    parser.add_argument("-d", "--dataset", required=True, help="Path to the dataset file.")
    parser.add_argument("-r", "--rounds", type=int, default=5, help="Number of rounds in the discussion.")
    parser.add_argument("-t", "--type", choices= ["SciFi-Dis", "SciFi-Teacher", "SciFi-Review"], help="Type of task to run.")
    # parser.add_argument("-e", "--eval_mode", action="store_true", default=False, help="Run in evaluation mode.")
    parser.add_argument("-p", "--prompt", type = int, default = 1, help = "Prompt Test")
    args = parser.parse_args()
    
    if args.type == "SciFi-Dis":
        agents_config = LLM_Discussion_SciFi.load_config(args.config)
        discussion_runner = LLM_Discussion_SciFi(agents_config, args.dataset, args.rounds, args.type, args.prompt)
    elif args.type == "SciFi-Teacher":
        agents_config = LLM_Teacher_SciFi.load_config(args.config)
        discussion_runner = LLM_Teacher_SciFi(agents_config, args.dataset, args.rounds, args.type, args.prompt)
    elif args.type == "SciFi-Review":
        agents_config = LLM_Review_SciFi.load_config(args.config)
        discussion_runner = LLM_Review_SciFi(agents_config, args.dataset, args.rounds, args.type, args.prompt)
    print('start session')
    discussion_output = discussion_runner.run()
    print('end session')

if __name__ == "__main__":
    main()
