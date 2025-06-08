-- Neo-tree is a Neovim plugin to browse the file system
-- https://github.com/nvim-neo-tree/neo-tree.nvim

return {
  'nvim-neo-tree/neo-tree.nvim',
  version = '*',
  dependencies = {
    'nvim-lua/plenary.nvim',
    'nvim-tree/nvim-web-devicons', -- not strictly required, but recommended
    'MunifTanjim/nui.nvim',
  },
  lazy = false,
  keys = {
    { '\\', ':Neotree reveal<CR>', desc = 'NeoTree reveal', silent = true },
  },
  ---@module "neo-tree"
  ---@type neotree.Config?
  opts = {
    -- enalbe close when window is closed
    close_if_last_window = true,
    enable_git_status = true,
    enable_diagnostics = true,
    window = {
      position = 'right', -- Position of the file explorer
    },
    filesystem = {
      follow_current_file = { enabled = true }, -- Auto-follow current file
      use_libuv_file_watcher = true, -- Auto-refresh
      window = {
        mappings = {
          ['\\'] = 'close_window',
        },
      },
    },
  },
}
