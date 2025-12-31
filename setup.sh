#!/bin/bash

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Setting up on $MACHINE..."

# vim
if [ -d .vim ]; then
    rsync -avr .vim   $HOME
fi
if [ -f .vimrc ]; then
    rsync -av  .vimrc $HOME
fi

# tmux
if [ -f .tmux.conf ]; then
    rsync -av  .tmux.conf $HOME
    if command -v tmux &> /dev/null; then
        tmux source-file ~/.tmux.conf 2>/dev/null || true
    fi
fi

# neovim
if [ -d nvim ]; then
    rsync -avr nvim $HOME/.config
fi

# tg-notify (install to ~/.local/bin)
mkdir -p $HOME/.local/bin
cp tg-notify.sh $HOME/.local/bin/tg-notify
chmod +x $HOME/.local/bin/tg-notify

# Detect current shell and appropriate rc file
detect_shell_rc() {
    if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "$(which zsh 2>/dev/null)" ]; then
        echo "$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ] || [ "$SHELL" = "$(which bash 2>/dev/null)" ]; then
        echo "$HOME/.bashrc"
    else
        # Default to shell name
        echo "$HOME/.$(basename $SHELL)rc"
    fi
}

# Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "Adding ~/.local/bin to PATH"

    PATH_EXPORT='export PATH="$HOME/.local/bin:$PATH"'

    # Add to both bashrc and zshrc if they exist or are likely to be used
    if [ "$MACHINE" = "Mac" ]; then
        # Mac uses zsh by default since Catalina
        if [ -f "$HOME/.zshrc" ] || [ "$SHELL" = "$(which zsh 2>/dev/null)" ]; then
            if ! grep -q '.local/bin' "$HOME/.zshrc" 2>/dev/null; then
                echo "$PATH_EXPORT" >> "$HOME/.zshrc"
                echo "  Updated ~/.zshrc"
            fi
        fi
        # Also add to bash profile for bash users on Mac
        if [ -f "$HOME/.bash_profile" ]; then
            if ! grep -q '.local/bin' "$HOME/.bash_profile" 2>/dev/null; then
                echo "$PATH_EXPORT" >> "$HOME/.bash_profile"
                echo "  Updated ~/.bash_profile"
            fi
        elif [ -f "$HOME/.bashrc" ]; then
            if ! grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null; then
                echo "$PATH_EXPORT" >> "$HOME/.bashrc"
                echo "  Updated ~/.bashrc"
            fi
        fi
    else
        # Linux typically uses bashrc
        if [ -f "$HOME/.bashrc" ]; then
            if ! grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null; then
                echo "$PATH_EXPORT" >> "$HOME/.bashrc"
                echo "  Updated ~/.bashrc"
            fi
        fi
        if [ -f "$HOME/.zshrc" ]; then
            if ! grep -q '.local/bin' "$HOME/.zshrc" 2>/dev/null; then
                echo "$PATH_EXPORT" >> "$HOME/.zshrc"
                echo "  Updated ~/.zshrc"
            fi
        fi
    fi

    echo "  Please run: source ~/.zshrc (or ~/.bashrc)"
fi

# Create .env template if it doesn't exist
if [ ! -f "$HOME/.tg-notify.env" ]; then
    cat > $HOME/.tg-notify.env << 'EOF'
# Telegram Bot Configuration
# Get bot token from @BotFather on Telegram
# Get chat ID from @userinfobot on Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
EOF
    echo "Created ~/.tg-notify.env - please edit it with your bot credentials"
fi

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. Edit ~/.tg-notify.env with your Telegram bot credentials"
echo "  2. Reload your shell: source ~/.zshrc (or ~/.bashrc)"
echo "  3. Test: tg-notify 'Hello from $(hostname)'"

