#!/bin/bash

# Telegram Notification Script
# Usage:
#   tg-notify "Your message here"
#   tg-notify "exp-name" command args...

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
    echo "[tg-notify] Error: Configuration file not found"
    echo "[tg-notify] Please create ~/.tg-notify.env with:"
    echo "[tg-notify]   TELEGRAM_BOT_TOKEN=your_bot_token_here"
    echo "[tg-notify]   TELEGRAM_CHAT_ID=your_chat_id_here"
    exit 1
fi

# Check if variables are set
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "[tg-notify] Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in config file"
    exit 1
fi

# Function to send notification
send_notification() {
    local message="$1"
    local hostname=$(hostname)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local full_message="[${hostname}] ${timestamp}
${message}"

    local response=$(curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        -d text="${full_message}" \
        -d parse_mode="HTML")

    if echo "$response" | grep -q '"ok":true'; then
        echo "[tg-notify] Notification sent successfully"
        return 0
    else
        echo "[tg-notify] Failed to send notification"
        echo "[tg-notify] Response: $response"
        return 1
    fi
}

# Check if we're in wrapper mode (label + command) or message mode
if [ $# -ge 2 ]; then
    # Wrapper mode: first arg is label, rest is command
    LABEL="$1"
    shift

    echo "[tg-notify] Running: $@"
    echo "[tg-notify] Label: $LABEL"
    echo ""

    # Run the command
    "$@"
    EXIT_CODE=$?

    # Send notification based on exit code
    if [ $EXIT_CODE -eq 0 ]; then
        send_notification "✓ Success: $LABEL"
    else
        send_notification "✗ Failed (Exit Code: $EXIT_CODE): $LABEL"
    fi

    exit $EXIT_CODE

elif [ $# -eq 1 ]; then
    # Message mode: just send the message
    send_notification "$1"
    exit $?

else
    # Read from stdin
    MESSAGE=$(cat)
    send_notification "$MESSAGE"
    exit $?
fi
