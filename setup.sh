#!/bin/bash

rsync -avr .vim       $HOME
rsync -av  .vimrc     $HOME
rsync -av  .tmux.conf $HOME


tmux source-file ~/.tmux.conf
