import argparse

BATCH_SIZE = 1

#DATA_PATH = "./data/"



def get_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    
    parser.add_argument("--task", type=str, default="1to1", help="task of this training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training. ")
    parser.add_argument("--weight", type=float, default=1, help="Weight for forward loss. ")
    
    
    return parser
