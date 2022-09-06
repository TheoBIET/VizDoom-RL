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
        game = Train(level_name)
        is_curriculum, n_difficulties = game.get_difficuties()
        
        if not is_curriculum:
            return Train(level_name).start()
        
        # If a difficulty is needed, request them to the user
        print(SELECT_A_DIFFICULTY.format(n_difficulties))
        difficulty_choice = typer.prompt(SELECT_AN_OPTION, default=1)
        
        if difficulty_choice > n_difficulties:
            raise ValueError(INVALID_CHOICE)
        
        Train(level_name).start(difficulty=difficulty_choice)
    
    else:
        raise ValueError(INVALID_CHOICE)

if __name__ == "__main__":
    typer.run(main)