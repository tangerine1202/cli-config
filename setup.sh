#!/bin/bash
# vim
rsync -avr .vim   $HOME
rsync -av  .vimrc $HOME

# tmux
rsync -av  .tmux.conf $HOME
tmux source-file ~/.tmux.conf

# neovim
rsync -avr nvim $HOME/.config

