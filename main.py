import typer
from rich import print
from rich.table import Table

from utils.constants.cli import *
from utils.constants.constants import *
from utils.train import Train
from utils.play import Play

def request_level():
    # Show all levels and ask for the one to train
    levels = '\n'.join([f'{i+1}. {option}' for i, option in enumerate(GAME_LEVELS)])
    print(TRAIN_SELECTION.format(levels))
    choice = typer.prompt(SELECT_AN_OPTION, default=1)
    print(choice)
    
    # Check if the choice is valid
    if choice > len(GAME_LEVELS):
        raise ValueError(INVALID_CHOICE)
    else:
        return choice
    
def main():
    print(WELCOME)
    choice = typer.prompt(SELECT_AN_OPTION, default=1)
    print(choice)

    if choice == PLAY:
        level_name = GAME_LEVELS[ request_level() - 1]
        Play(level_name).start()
    elif choice == TRAIN:
        level_name = GAME_LEVELS[ request_level() - 1]
        Train(level_name).start()
    
    else:
        raise ValueError(INVALID_CHOICE)

if __name__ == "__main__":
    typer.run(main)