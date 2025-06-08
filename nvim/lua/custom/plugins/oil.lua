return {
  'stevearc/oil.nvim',
  enabled = false,

  ---@module 'oil'
  ---@type oil.SetupOpts
  opts = {
    lsp_file_methods = {
      -- Set to true to autosave buffers that are updated with LSP willRenameFiles
      -- Set to "unmodified" to only save unmodified buffers
      autosave_changes = true,
    },
    -- Set to true to watch the filesystem for changes and reload oil
    watch_for_changes = false,
  },
  -- Optional dependencies
  -- dependencies = { { 'echasnovski/mini.icons', opts = {} } },
  dependencies = { 'nvim-tree/nvim-web-devicons' }, -- use if you prefer nvim-web-devicons
  -- Lazy loading is not recommended because it is very tricky to make it work correctly in all situations.
  lazy = false,
}
