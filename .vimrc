set number

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

" statusline
" set laststatus=2
" set statusline+=%#StatusLine#
" set statusline+=%{StatuslineGit()}
" set statusline+=%F
" 
" set statusline+=%=
" set statusline+=\ %{&fileencoding?&fileencoding:&encoding}
" set statusline+=\ %y
" set statusline+=%r
" 
" function! GitBranch()
"   return system("git rev-parse --abbrev-ref HEAD 2>/dev/null | tr -d '\n'")
" endfunction
" 
" function! StatuslineGit()
"   let l:branchname = GitBranch()
"   return strlen(l:branchname) > 0 ? '['.l:branchname.'] ' : ''
" endfunction
"
"
" ---- color scheme -----
"
" onedark: https://github.com/joshdick/onedark.vim
"
"NOTE: only un-commented when outside tmux
"Use 24-bit (true-color) mode in Vim/Neovim when outside tmux.
"If you're using tmux version 2.2 or later, you can remove the outermost $TMUX check and use tmux's 24-bit color support
"(see < http://sunaku.github.io/tmux-24bit-color.html#usage > for more information.)
" if (empty($TMUX))
"   if (has("nvim"))
"     "For Neovim 0.1.3 and 0.1.4 < https://github.com/neovim/neovim/pull/2198 >
"     let $NVIM_TUI_ENABLE_TRUE_COLOR=1
"   endif
"   "For Neovim > 0.1.5 and Vim > patch 7.4.1799 < https://github.com/vim/vim/commit/61be73bb0f965a895bfb064ea3e55476ac175162 >
"   "Based on Vim patch 7.4.1770 (`guicolors` option) < https://github.com/vim/vim/commit/8a633e3427b47286869aa4b96f2bfc1fe65b25cd >
"   " < https://github.com/neovim/neovim/wiki/Following-HEAD#20160511 >
"   if (has("termguicolors"))
"     set termguicolors
"   endif
" endif
set termguicolors

" Remove background color (only when running in terminals$)
" if (has("autocmd") && !has("gui_running"))$
"   augroup colorset$
"     autocmd!$
"     let s:white = { "gui": "#ABB2BF", "cterm": "145", "cterm16" : "7" }$
"     autocmd ColorScheme * call onedark#set_highlight("Normal", { "fg": s:white }) " `bg` will not be styled since there is n    o `bg` setting$
"   augroup END$
" endif$

" Place the following lines *after* the onedark configuration
syntax on
colorscheme onedark
