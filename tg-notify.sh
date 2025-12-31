#!/bin/bash

# Telegram Notification Script
# Usage: tg-notify "Your message here"

# Configuration file path (supports both .env and .conf format)
ENV_FILE="${HOME}/.tg-notify.env"
CONF_FILE="${HOME}/.tg-notify.conf"

# Function to load .env file (handles export statements and comments)
load_env() {
    local env_file="$1"
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue

        # Remove 'export ' prefix if present
        key="${key#export }"
        key="${key// /}"  # Remove spaces

        # Remove quotes from value
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"

        # Export the variable
        export "$key=$value"
    done < "$env_file"
}

# Check for config file (prefer .env format)
if [ -f "$ENV_FILE" ]; then
    load_env "$ENV_FILE"
elif [ -f "$CONF_FILE" ]; then
    source "$CONF_FILE"
else
    echo "Error: Configuration file not found"
    echo "Please create ~/.tg-notify.env with:"
    echo "TELEGRAM_BOT_TOKEN=your_bot_token_here"
    echo "TELEGRAM_CHAT_ID=your_chat_id_here"
    exit 1
fi

# Check if variables are set
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in config file"
    exit 1
fi

# Get message from argument or stdin
if [ -n "$1" ]; then
    MESSAGE="$1"
else
    MESSAGE=$(cat)
fi

# Add hostname and timestamp
HOSTNAME=$(hostname)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
FULL_MESSAGE="[${HOSTNAME}] ${TIMESTAMP}
${MESSAGE}"

# Send message via Telegram API
RESPONSE=$(curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" \
    -d text="${FULL_MESSAGE}" \
    -d parse_mode="HTML")

# Check if successful
if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "Notification sent successfully"
    exit 0
else
    echo "Failed to send notification"
    echo "Response: $RESPONSE"
    exit 1
fi
