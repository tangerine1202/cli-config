#!/usr/bin/env bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVIM_CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/nvim"
NVIM_INSTALLED_BY_SCRIPT=false

# Function to check GLIBC version
check_glibc_version() {
    # Get GLIBC version
    if command -v ldd &> /dev/null; then
        GLIBC_VERSION=$(ldd --version 2>/dev/null | head -n 1 | grep -oP '\d+\.\d+' | head -n 1)
    else
        # Fallback: try to get version from libc.so.6
        if [ -f /lib/x86_64-linux-gnu/libc.so.6 ]; then
            GLIBC_VERSION=$(/lib/x86_64-linux-gnu/libc.so.6 2>&1 | grep -oP 'release version \K\d+\.\d+' | head -n 1)
        elif [ -f /lib64/libc.so.6 ]; then
            GLIBC_VERSION=$(/lib64/libc.so.6 2>&1 | grep -oP 'release version \K\d+\.\d+' | head -n 1)
        fi
    fi

    # Check if GLIBC is too old for AppImage (requires 2.31+)
    if [ -n "$GLIBC_VERSION" ]; then
        GLIBC_MAJOR=$(echo "$GLIBC_VERSION" | cut -d. -f1)
        GLIBC_MINOR=$(echo "$GLIBC_VERSION" | cut -d. -f2)

        if [ "$GLIBC_MAJOR" -lt 2 ] || ([ "$GLIBC_MAJOR" -eq 2 ] && [ "$GLIBC_MINOR" -lt 31 ]); then
            return 1  # GLIBC too old
        fi
    fi
    return 0  # GLIBC is new enough or unknown
}

# Function to install Neovim via AppImage
install_neovim_appimage() {
    echo -e "\n${BLUE}=== Installing Neovim via AppImage ===${NC}\n"

    # Check GLIBC version first
    if ! check_glibc_version; then
        echo -e "${YELLOW}Warning: Your system has GLIBC ${GLIBC_VERSION}${NC}"
        echo "The Neovim AppImage requires GLIBC 2.31 or newer."
        echo ""
        echo -e "${BLUE}Alternative installation options:${NC}"
        echo "  1. Use tarball from neovim-releases (for old GLIBC):"
        echo "     https://github.com/neovim/neovim-releases/releases"
        echo ""
        echo "  2. Build from source:"
        echo "     git clone https://github.com/neovim/neovim.git"
        echo "     cd neovim && make CMAKE_BUILD_TYPE=RelWithDebInfo"
        echo ""
        echo "  3. Use system package manager (may have older version):"
        echo "     sudo apt install neovim"
        echo ""
        read -p "Try AppImage anyway? It may not work. (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi

    # Check for download tools
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        echo -e "${RED}Error: Neither curl nor wget is available${NC}"
        echo "Please install curl or wget first"
        exit 1
    fi

    # Create ~/.local/bin if it doesn't exist
    mkdir -p "$HOME/.local/bin"

    NVIM_PATH="$HOME/.local/bin/nvim"
    APPIMAGE_URL="https://github.com/neovim/neovim/releases/latest/download/nvim.appimage"

    echo "Downloading Neovim AppImage..."

    # Download AppImage
    if command -v curl &> /dev/null; then
        curl -L -o "$NVIM_PATH" "$APPIMAGE_URL" || {
            echo -e "${RED}Failed to download Neovim AppImage${NC}"
            exit 1
        }
    else
        wget -O "$NVIM_PATH" "$APPIMAGE_URL" || {
            echo -e "${RED}Failed to download Neovim AppImage${NC}"
            exit 1
        }
    fi

    # Validate the download is actually a binary (not HTML error page)
    FILE_TYPE=$(file -b "$NVIM_PATH" 2>/dev/null || echo "unknown")
    if [[ ! "$FILE_TYPE" =~ (ELF|executable|AppImage) ]]; then
        echo -e "${RED}Error: Downloaded file is not a valid AppImage${NC}"
        echo "File type detected: $FILE_TYPE"
        echo ""
        echo "This usually means:"
        echo "  1. GitHub releases URL has changed"
        echo "  2. Network/proxy issues"
        echo "  3. Download was interrupted"
        echo ""
        echo "Cleaning up invalid download..."
        rm -f "$NVIM_PATH"
        echo "Please try again or install manually from:"
        echo "  https://github.com/neovim/neovim/releases"
        exit 1
    fi

    # Make executable
    chmod u+x "$NVIM_PATH"

    # Verify installation
    if [ -x "$NVIM_PATH" ]; then
        echo -e "${GREEN}✓ Neovim AppImage downloaded successfully${NC}"

        # Test if nvim works
        if "$NVIM_PATH" --version &> /dev/null; then
            NVIM_INSTALLED_BY_SCRIPT=true
            NVIM_VERSION=$("$NVIM_PATH" --version | head -n 1)
            echo -e "${GREEN}✓ Neovim is working: ${NVIM_VERSION}${NC}"
            echo ""

            # Check if ~/.local/bin is in PATH
            if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
                echo -e "${YELLOW}Note: ~/.local/bin is not in your PATH${NC}"
                echo "Add the following line to your ~/.bashrc or ~/.zshrc:"
                echo -e "${BLUE}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
                echo ""
                echo "Then reload your shell with: source ~/.bashrc (or ~/.zshrc)"
                echo ""
            fi
        else
            echo -e "${YELLOW}Warning: Neovim was installed but may require FUSE to run${NC}"
            echo "If it doesn't work, extract the AppImage:"
            echo "  $NVIM_PATH --appimage-extract"
            echo "  mv squashfs-root ~/.local/nvim"
            echo "  ln -sf ~/.local/nvim/AppRun ~/.local/bin/nvim"
            echo ""
        fi
    else
        echo -e "${RED}Failed to install Neovim AppImage${NC}"
        exit 1
    fi
}

echo -e "${BLUE}=== Neovim Configuration Setup ===${NC}\n"

# Check if nvim is installed and working
NVIM_WORKING=false
if command -v nvim &> /dev/null; then
    # Check if nvim actually works
    if NVIM_VERSION=$(nvim --version 2>/dev/null | head -n 1); then
        if [ -n "$NVIM_VERSION" ]; then
            NVIM_WORKING=true
            echo -e "${GREEN}Found: ${NVIM_VERSION}${NC}\n"
        fi
    fi
fi

if [ "$NVIM_WORKING" = false ]; then
    if command -v nvim &> /dev/null; then
        # nvim exists but is corrupted
        echo -e "${YELLOW}Neovim is installed but appears to be corrupted${NC}"
        NVIM_PATH=$(command -v nvim)
        echo "Location: $NVIM_PATH"
        read -p "Would you like to reinstall Neovim via AppImage? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Remove corrupted installation if it's in ~/.local/bin
            if [[ "$NVIM_PATH" == "$HOME/.local/bin/nvim" ]]; then
                echo "Removing corrupted installation..."
                rm -f "$NVIM_PATH"
            fi
            install_neovim_appimage
        else
            echo -e "${RED}Cannot proceed with corrupted Neovim installation${NC}"
            echo "Please fix or remove: $NVIM_PATH"
            exit 1
        fi
    else
        # nvim doesn't exist
        echo -e "${YELLOW}Neovim is not installed${NC}"
        read -p "Would you like to install Neovim via AppImage? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_neovim_appimage
        else
            echo -e "${BLUE}You can install Neovim manually from:${NC}"
            echo "  https://github.com/neovim/neovim/wiki/Installing-Neovim"
            exit 0
        fi
    fi
fi

# Check dependencies (non-blocking)
echo -e "${BLUE}Checking dependencies...${NC}"
MISSING_DEPS=()

check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        MISSING_DEPS+=("$1")
    else
        echo -e "  ${GREEN}✓${NC} $1"
    fi
}

check_dependency "git"
check_dependency "make"
check_dependency "rg"  # ripgrep
check_dependency "fd"  # fd-find

# Check for clipboard support (required on Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xclip &> /dev/null || command -v xsel &> /dev/null || command -v wl-copy &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} clipboard (xclip/xsel/wl-clipboard)"
    else
        echo -e "  ${RED}✗${NC} clipboard tool (xclip/xsel/wl-clipboard)"
        MISSING_DEPS+=("xclip or xsel or wl-clipboard")
    fi
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo -e "\n${YELLOW}Warning: Missing optional dependencies:${NC}"
    for dep in "${MISSING_DEPS[@]}"; do
        echo -e "  ${YELLOW}✗${NC} $dep"
    done
    echo -e "\nSome features may not work without these dependencies."
    echo "You can install them later if needed."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Handle existing config
if [ -e "$NVIM_CONFIG_DIR" ]; then
    # Backup existing config
    BACKUP_DIR="${NVIM_CONFIG_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}Backing up existing config to: ${BACKUP_DIR}${NC}"
    mv "$NVIM_CONFIG_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}Backup created successfully${NC}\n"
fi

# Copy configuration files
echo -e "${BLUE}Copying configuration files...${NC}"
echo "  From: $SCRIPT_DIR"
echo "  To:   $NVIM_CONFIG_DIR"

# Create parent directory if needed
mkdir -p "$(dirname "$NVIM_CONFIG_DIR")"

# Copy all files
if command -v rsync &> /dev/null; then
    # Use rsync if available (preserves permissions and is more efficient)
    rsync -a --exclude='.git' --exclude='.gitignore' --exclude='setup.sh' --exclude='README.md' "$SCRIPT_DIR/" "$NVIM_CONFIG_DIR/"
else
    # Fallback to cp
    cp -r "$SCRIPT_DIR/." "$NVIM_CONFIG_DIR/"
    # Remove unwanted files
    rm -rf "$NVIM_CONFIG_DIR/.git" "$NVIM_CONFIG_DIR/.gitignore" "$NVIM_CONFIG_DIR/setup.sh" "$NVIM_CONFIG_DIR/README.md"
fi

if [ -d "$NVIM_CONFIG_DIR" ]; then
    echo -e "${GREEN}✓ Configuration files copied successfully${NC}\n"
else
    echo -e "${RED}✗ Failed to copy configuration files${NC}"
    exit 1
fi

# Clean up old Neovim data (optional)
NVIM_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/nvim"
NVIM_STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/nvim"
NVIM_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/nvim"

echo -e "${BLUE}Neovim data directories:${NC}"
echo "  Data:  $NVIM_DATA_DIR"
echo "  State: $NVIM_STATE_DIR"
echo "  Cache: $NVIM_CACHE_DIR"
echo ""
read -p "Do you want to clean these directories? This will remove plugins and data. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    [ -d "$NVIM_DATA_DIR" ] && rm -rf "$NVIM_DATA_DIR" && echo -e "${GREEN}✓ Cleaned data directory${NC}"
    [ -d "$NVIM_STATE_DIR" ] && rm -rf "$NVIM_STATE_DIR" && echo -e "${GREEN}✓ Cleaned state directory${NC}"
    [ -d "$NVIM_CACHE_DIR" ] && rm -rf "$NVIM_CACHE_DIR" && echo -e "${GREEN}✓ Cleaned cache directory${NC}"
else
    echo "Keeping existing Neovim data directories"
fi

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}\n"

if [ "$NVIM_INSTALLED_BY_SCRIPT" = true ]; then
    echo -e "${YELLOW}Important: Neovim was installed to ~/.local/bin/nvim${NC}"
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo "You need to add ~/.local/bin to your PATH:"
        echo -e "  ${BLUE}echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc${NC}"
        echo "  source ~/.bashrc"
        echo ""
        echo "Or for zsh:"
        echo -e "  ${BLUE}echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc${NC}"
        echo "  source ~/.zshrc"
        echo ""
    fi
fi

echo "Next steps:"
echo "  1. Start Neovim: nvim"
echo "  2. Lazy will automatically install plugins"
echo "  3. Run :checkhealth to verify installation"
echo ""
echo "For more information, see README.md"
