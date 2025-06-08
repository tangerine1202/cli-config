---@diagnostic disable: unused-local
-- If you want to see what colorschemes are already installed, you can use `:Telescope colorscheme`.

local scheme = 'tokyonight'

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
      vim.cmd.colorscheme 'tokyonight'
    end
  end,
}

local onedark = {
  -- https://github.com/olimorris/onedarkpro.nvim
  'olimorris/onedarkpro.nvim',
  priority = 1000,
  opts = {
    colors = {
      cursorline = "require('onedarkpro.helpers').darken('gray', 17)",
    },
    options = {
      cursorline = true,
    },
  },
  config = function(_, opts)
    require('onedarkpro').setup(opts)
    if scheme == 'onedark' then
      vim.cmd.colorscheme 'onedark'
    end
  end,
}

return {
  tokyonight,
  onedark,
}
