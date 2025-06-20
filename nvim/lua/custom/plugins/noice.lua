return {
  'folke/noice.nvim',
  enabled = true,
  event = 'VeryLazy',
  dependencies = {
    -- if you lazy-load any plugin below, make sure to add proper `module="..."` entries
    'MunifTanjim/nui.nvim',
    -- OPTIONAL:

    -- If not available, we use `mini` as the fallback
    {
      'rcarriga/nvim-notify',
      -- move the notify to the bottom of the screen
      -- https://github.com/LazyVim/LazyVim/discussions/2481#discussioncomment-8337157
      opts = { top_down = false },
    },
  },
  opts = {
    lsp = {
      -- override markdown rendering so that **cmp** and other plugins use **Treesitter**
      override = {
        ['vim.lsp.util.convert_input_to_markdown_lines'] = true,
        ['vim.lsp.util.stylize_markdown'] = true,
        ['cmp.entry.get_documentation'] = true, -- requires hrsh7th/nvim-cmp
      },
    },
    -- you can enable a preset for easier configuration
    presets = {
      bottom_search = true, -- use a classic bottom cmdline for search
      long_message_to_split = true, -- long messages will be sent to a split
      inc_rename = false, -- enables an input dialog for inc-rename.nvim
      lsp_doc_border = false, -- add a border to hover docs and signature help
      command_palette = { -- position the cmdline and popupmenu together
        -- position the cmdline at the center
        -- ref: https://github.com/LazyVim/LazyVim/discussions/2481#discussioncomment-8345232
        views = {
          cmdline_popup = {
            position = {
              row = '50%',
              col = '50%',
            },
            size = {
              min_width = 60,
              width = 'auto',
              height = 'auto',
            },
          },
          cmdline_popupmenu = {
            position = {
              row = '67%',
              col = '50%',
            },
          },
        },
      },
    },
  },
}
