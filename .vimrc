set number

set tabstop=2                                   " a tab is how many spaces
set shiftwidth=2                                " number of spaces to use for autoindenting
set autoindent                                  " always set autoindenting on
filetype indent on

set smarttab            " insert tabs on the start of a line according to shiftwidth, not tabstop

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

" Remove background color (only when running in terminals$)
" if (has("autocmd") && !has("gui_running"))$
"   augroup colorset$
"     autocmd!$
"     let s:white = { "gui": "#ABB2BF", "cterm": "145", "cterm16" : "7" }$
"     autocmd ColorScheme * call onedark#set_highlight("Normal", { "fg": s:white }) " `bg` will not be styled since there is n    o `bg` setting$
"   augroup END$
" endif$

" place the following lines *after* the onedark configuration
syntax on
colorscheme onedark
