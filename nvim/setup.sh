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

echo -e "${BLUE}=== Neovim Configuration Setup ===${NC}\n"

# Check if nvim is installed
if ! command -v nvim &> /dev/null; then
    echo -e "${RED}Error: Neovim is not installed${NC}"
    echo "Please install Neovim first: https://github.com/neovim/neovim/wiki/Installing-Neovim"
    exit 1
fi

NVIM_VERSION=$(nvim --version | head -n 1)
echo -e "${GREEN}Found: ${NVIM_VERSION}${NC}\n"

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
echo "Next steps:"
echo "  1. Start Neovim: nvim"
echo "  2. Lazy will automatically install plugins"
echo "  3. Run :checkhealth to verify installation"
echo ""
echo "For more information, see README.md"
