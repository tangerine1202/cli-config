set relativenumber

set tabstop=2                                   " a tab is how many spaces
set softtabstop=2
set shiftwidth=2                                " number of spaces to use for autoindenting
set expandtab
set smarttab            " insert tabs on the start of a line according to shiftwidth, not tabstop

set autoindent                                  " always set autoindenting on
filetype indent on

set scrolloff=3         " keep 4 lines off the edges of the screen when scrolling

set hlsearch            " highlight search terms
set incsearch           " show search matches as you type

" white space character
set list
set listchars=eol:$,tab::\ ,trail:.,extends:>,precedes:<,nbsp:_


"
" ---- color scheme -----
"
" onedark: https://github.com/joshdick/onedark.vim

" Enable true color support
set termguicolors

" Place the following lines *after* the onedark configuration
syntax on
colorscheme onedark

" ----- .bashrc useful command -----
" Auto start tmux when connecting via SSH:
" if [[ -n "$PS1" ]] && [[ -z "$TMUX" ]] && [[ -n "$SSH_CONNECTION" ]]; then
"   tmux attach-session -t ssh_tmux || tmux new-session -s ssh_tmux
" fi

