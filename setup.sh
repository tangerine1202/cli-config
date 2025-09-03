#!/bin/bash
# vim
rsync -avr .vim   $HOME
rsync -av  .vimrc $HOME

# tmux
rsync -av  .tmux.conf $HOME
# Source tmux config only if tmux is running
if tmux info &>/dev/null; then
    tmux source-file ~/.tmux.conf
fi

# neovim
rsync -avr nvim $HOME/.config

