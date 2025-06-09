---@diagnostic disable: unused-local
-- If you want to see what colorschemes are already installed, you can use `:Telescope colorscheme`.

local scheme = 'onedark'

local tokyonight = {
  'folke/tokyonight.nvim',
  priority = 1000, -- Make sure to load this before all the other start plugins.
  opts = {
    style = 'night', -- 'storm' | 'day' | 'night' | 'moon'
    styles = {
      comments = { italic = false }, -- Disable italics in comments
    },
  },
  config = function(_, opts)
    require('tokyonight').setup(opts)
    if scheme == 'tokyonight' then
      vim.cmd.colorscheme(scheme)
    end
  end,
}

local onedark = {
  -- https://github.com/olimorris/onedarkpro.nvim
  'olimorris/onedarkpro.nvim',
  priority = 1000,
  opts = {
    style = '', -- 'vivid' | 'dark'
    colors = {
      dark = {
        cursorline = "require('onedarkpro.helpers').lighten('bg', 6)",
      },
      light = {
        cursorline = "require('onedarkpro.helpers').lighten('bg', 6)",
      },
    },
    highlights = {
      -- telescope
      TelescopePromptCounter = { fg = '${fg}' },
      TelescopePromptTitle = { fg = '${purple}' },
      TelescopePreviewTitle = { fg = '${green}' },
      TelescopeResultsTitle = { fg = '${yellow}' },
      TelescopeSelection = { bg = '${cursorline}' },
      TelescopeSelectionCaret = { fg = '${purple}', bg = '${cursorline}' },
      TelescopeMatching = { fg = '${blue}' },
      -- mini statue line
      MiniStatuslineModeNormal = { fg = '${bg}', bg = '${blue}', bold = true },
      MiniStatuslineModeInsert = { fg = '${bg}', bg = '${green}', bold = true },
      MiniStatuslineModeVisual = { fg = '${bg}', bg = '${purple}', bold = true },
      MiniStatuslineModeReplace = { fg = '${bg}', bg = '${yellow}', bold = true },
      MiniStatuslineModeCommand = { fg = '${bg}', bg = '${fg}', bold = true },
      MiniStatuslineModeOther = { fg = '${fg}', bg = '${bg}', bold = true },
      StatusLine = { fg = '${fg}', bg = '${cursorline}' },
      StatusLineNC = { fg = '${fg}', bg = '${bg}' },
    },
    options = {
      cursorline = true,
    },
  },
  config = function(_, opts)
    require('onedarkpro').setup(opts)
    if scheme == 'onedark' or scheme == 'onelight' or scheme == 'vaporwave' then
      vim.cmd.colorscheme(scheme .. ((opts.style ~= '' and ('_' .. opts.style)) or ''))
    end
  end,
}

return {
  tokyonight,
  onedark,
}
